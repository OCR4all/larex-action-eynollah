# SPDX-License-Identifier: MIT
import asyncio
import logging
import re
import tempfile
from collections import deque
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from larex_actions import ActionContext, PageResultRejected
from larex_actions.models import ActionInput, ActionPage

from .config import Settings
from .models import missing_models, required_models
from .parameters import InvalidParameters, LayoutOptions, parse_parameters
from .pool import WorkerCrashed, WorkerLoadError, WorkerPool, WorkerProcess, WorkerStopped

CANCEL_POLL_SECONDS = 1.0
QUEUE_POLL_SECONDS = 5.0
MAX_CONSECUTIVE_CRASHES = 3
SHUTDOWN_REPORT_SECONDS = 10

logger = logging.getLogger(__name__)


class RunFailed(Exception):
    pass


@dataclass(frozen=True)
class Outcome:
    status: Literal['SUCCEEDED', 'FAILED', 'SKIPPED']
    reason: str = ''
    xml: bytes | None = None
    seconds: float = 0


@dataclass
class Totals:
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0

    @property
    def settled(self) -> int:
        return self.succeeded + self.failed + self.skipped

    def count(self, outcome: Outcome) -> None:
        name = outcome.status.lower()
        setattr(self, name, getattr(self, name) + 1)

    def describe(self, total: int) -> str:
        parts = [f'Segmented {self.succeeded} of {total} page(s)']
        if self.failed:
            parts.append(f'{self.failed} failed')
        if self.skipped:
            parts.append(f'{self.skipped} without any regions')
        return ', '.join(parts)


class Runner:
    def __init__(self, settings: Settings, pool: WorkerPool):
        self._settings = settings
        self._pool = pool
        self._slots = asyncio.Semaphore(settings.ACTION_MAX_CONCURRENT_RUNS)
        self._active: set[asyncio.Task] = set()

    async def __call__(self, ctx: ActionContext) -> None:
        task = asyncio.current_task()
        self._active.add(task)
        try:
            await self._run(ctx)
        except asyncio.CancelledError:
            logger.warning('Run %s was interrupted by a shutdown', ctx.run_id)
            with suppress(Exception):
                await asyncio.wait_for(ctx.fail('The processor is shutting down'), SHUTDOWN_REPORT_SECONDS)
            raise
        finally:
            self._active.discard(task)

    async def close(self) -> None:
        for task in self._active:
            task.cancel()
        if self._active:
            await asyncio.wait(self._active, timeout=SHUTDOWN_REPORT_SECONDS + 5)
        await self._pool.close()

    async def _run(self, ctx: ActionContext) -> None:
        try:
            action_input = await ctx.pull_input()
            options = self._validate(action_input)
            async with self._run_slot(ctx):
                message = await PageRun(ctx, action_input, options, self._settings, self._pool).execute()
            await ctx.complete(message=message)
        except (RunFailed, InvalidParameters, WorkerLoadError) as error:
            logger.error('Run %s failed: %s', ctx.run_id, error)
            await ctx.fail(str(error))

    def _validate(self, action_input: ActionInput) -> LayoutOptions:
        if not action_input.pages:
            raise RunFailed('The action input contains no pages')
        options = parse_parameters(action_input.parameters)
        missing = missing_models(self._settings.EYNOLLAH_MODEL_DIR, required_models(options))
        if missing:
            raise RunFailed(f'The processor is missing the models: {", ".join(missing)}')
        return options

    @asynccontextmanager
    async def _run_slot(self, ctx: ActionContext) -> AsyncIterator[None]:
        acquiring = asyncio.ensure_future(self._slots.acquire())
        try:
            while not (await asyncio.wait([acquiring], timeout=QUEUE_POLL_SECONDS))[0]:
                await ctx.heartbeat(status_message='Waiting for another run to finish', raise_on_cancel=True)
            yield
        finally:
            acquiring.cancel()
            await asyncio.gather(acquiring, return_exceptions=True)
            if not acquiring.cancelled():
                self._slots.release()


class PageRun:
    def __init__(
        self,
        ctx: ActionContext,
        action_input: ActionInput,
        options: LayoutOptions,
        settings: Settings,
        pool: WorkerPool,
    ):
        self._ctx = ctx
        self._pages = action_input.pages
        self._options = options
        self._settings = settings
        self._pool = pool
        self._queue = deque(enumerate(self._pages, start=1))
        self._totals = Totals()
        self._worker: WorkerProcess | None = None
        self._worker_lock = asyncio.Lock()
        self._isolation_lock = asyncio.Lock()
        self._submit_lock = asyncio.Lock()
        self._permits = asyncio.Semaphore(1)
        self._slots = 1
        self._crashes = 0

    async def execute(self) -> str:
        try:
            await run_until_done(self._process_pages(), self._watch_cancellation())
        except BaseException:
            await self._discard_worker()
            raise
        if self._worker:
            await self._pool.release(self._worker)
        return self._totals.describe(len(self._pages))

    async def _watch_cancellation(self) -> None:
        while True:
            await self._ctx.check_cancelled()
            await asyncio.sleep(CANCEL_POLL_SECONDS)

    async def _process_pages(self) -> None:
        await self._ctx.heartbeat(0, 'Loading the models', raise_on_cancel=True)
        self._worker = await self._pool.acquire(self._options)
        self._slots = self._worker.page_workers
        self._permits = asyncio.Semaphore(self._slots)
        with tempfile.TemporaryDirectory(prefix='eynollah-') as directory:
            await all_of(*(self._consume(Path(directory)) for _ in range(self._slots + 1)))

    async def _consume(self, directory: Path) -> None:
        while self._queue:
            index, page = self._queue.popleft()
            outcome = await self._process_page(directory / str(index), page)
            await self._submit(index, page, outcome)

    async def _process_page(self, path: Path, page: ActionPage) -> Outcome:
        if not page.images:
            return Outcome('FAILED', 'The page has no image')
        image = page.images[0]
        try:
            try:
                await self._ctx.download_to_path(image, path)
            except (httpx.HTTPError, OSError) as error:
                logger.warning('Could not download the image of %s: %s', page.name, error)
                return Outcome('FAILED', f'The image could not be downloaded: {type(error).__name__}')
            return await self._segment(path, image.file_name)
        finally:
            path.unlink(missing_ok=True)

    async def _segment(self, path: Path, file_name: str) -> Outcome:
        isolated = False
        while True:
            async with self._isolation() if isolated else self._permits:
                worker = await self._current_worker()
                try:
                    return await self._request(worker, path, file_name)
                except WorkerStopped:
                    continue
                except WorkerCrashed as error:
                    if isolated or self._slots == 1:
                        return self._crashed(error)
                    isolated = True

    async def _request(self, worker: WorkerProcess, path: Path, file_name: str) -> Outcome:
        timeout = self._settings.ACTION_PROCESSING_TIMEOUT
        try:
            async with asyncio.timeout(timeout):
                reply = await worker.process(self._options, path, file_name)
        except TimeoutError:
            message = f'The page was not finished within {timeout} seconds'
            if self._settings.ACTION_TIMEOUT_MODE == 'fail_action':
                raise RunFailed(message) from None
            await self._pool.discard(worker)
            return Outcome('FAILED', message)
        if reply.status == 'broken':
            await self._pool.discard(worker)
            raise WorkerCrashed(f'The inference runtime failed: {reply.message}')
        self._crashes = 0
        if reply.status == 'failed':
            return Outcome('FAILED', reply.message)
        if reply.status == 'empty':
            return Outcome('SKIPPED', 'No regions were found, existing annotations were kept')
        return Outcome('SUCCEEDED', xml=reply.xml, seconds=reply.seconds)

    def _crashed(self, error: WorkerCrashed) -> Outcome:
        self._crashes += 1
        if self._crashes >= MAX_CONSECUTIVE_CRASHES:
            raise RunFailed(f'Stopped after {self._crashes} consecutive worker failures. {error}')
        return Outcome('FAILED', str(error))

    @asynccontextmanager
    async def _isolation(self) -> AsyncIterator[None]:
        acquired = 0
        async with self._isolation_lock:
            try:
                for _ in range(self._slots):
                    await self._permits.acquire()
                    acquired += 1
                yield
            finally:
                for _ in range(acquired):
                    self._permits.release()

    async def _current_worker(self) -> WorkerProcess:
        async with self._worker_lock:
            if not self._worker.alive:
                stale, self._worker = self._worker, None
                await self._pool.discard(stale)
            if self._worker is None:
                self._worker = await self._pool.acquire(self._options)
            return self._worker

    async def _discard_worker(self) -> None:
        if self._worker:
            await self._pool.discard(self._worker)

    async def _submit(self, index: int, page: ActionPage, outcome: Outcome) -> None:
        total = len(self._pages)
        label = f'[{index}/{total}] {page.name}'
        results = self._ctx.result_builder()
        if outcome.xml:
            results.add_xml_bytes(page_id=page.id, content=outcome.xml, file_name=result_file_name(page))
            message = f'{label}: segmented in {outcome.seconds:.1f} s'
        else:
            message = f'{label}: {outcome.reason}'
        async with self._submit_lock:
            try:
                await self._ctx.submit_page_results(
                    page.id, results, message, page_status=outcome.status, reason=outcome.reason or None
                )
            except PageResultRejected as error:
                outcome = Outcome('FAILED', f'LAREX rejected the result: {error}')
                message = f'{label}: {outcome.reason}'
            logger.log(logging.WARNING if outcome.status == 'FAILED' else logging.INFO, message)
            self._totals.count(outcome)
            await self._ctx.heartbeat(self._totals.settled * 100 // total, message, raise_on_cancel=True)


def result_file_name(page: ActionPage) -> str:
    stem = re.sub(r'[^A-Za-z0-9._-]+', '-', Path(page.name).stem).strip('._-')
    return f'{stem[:96] or "page"}.xml'


async def run_until_done(work: Coroutine[Any, Any, None], monitor: Coroutine[Any, Any, None]) -> None:
    tasks = [asyncio.create_task(monitor), asyncio.create_task(work)]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        raise_first_failure(tasks)
    finally:
        await cancel(tasks)


async def all_of(*work: Coroutine[Any, Any, None]) -> None:
    tasks = [asyncio.create_task(coroutine) for coroutine in work]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        raise_first_failure(tasks)
    finally:
        await cancel(tasks)


def raise_first_failure(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        if task.done():
            task.result()


async def cancel(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
