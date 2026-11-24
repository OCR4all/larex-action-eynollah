# SPDX-License-Identifier: MIT
import asyncio
import itertools
import logging
import multiprocessing
import threading
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from .parameters import LayoutOptions
from .worker import LoadRequest, PageRequest, Reply, Started, WorkerSettings, worker

STOP_TIMEOUT = 5

logger = logging.getLogger(__name__)


class WorkerGone(Exception):
    pass


class WorkerCrashed(WorkerGone):
    pass


class WorkerStopped(WorkerGone):
    pass


class WorkerLoadError(Exception):
    pass


class WorkerProcess:
    def __init__(self, settings: WorkerSettings, target: Callable = worker):
        self._loop = asyncio.get_running_loop()
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future[Reply]] = {}
        self._gone: WorkerGone | None = None
        self._stopping = False
        self._started: asyncio.Future[Started] = self._loop.create_future()
        self._started.add_done_callback(_retrieve_exception)
        self._exited = asyncio.Event()
        self._launching: asyncio.Future | None = None
        self._outstanding: set[int] = set()
        self._send_lock = threading.Lock()
        context = multiprocessing.get_context('spawn')
        self._connection, self._child_connection = context.Pipe()
        self._process = context.Process(
            target=target, args=(self._child_connection, settings), name='eynollah-worker', daemon=True
        )
        self.page_workers = 1

    @property
    def alive(self) -> bool:
        return self._gone is None and not self._stopping

    @property
    def in_flight(self) -> int:
        return len(self._outstanding)

    async def start(self, options: LayoutOptions, timeout: float | None) -> None:
        try:
            async with asyncio.timeout(timeout):
                self._launching = asyncio.ensure_future(asyncio.to_thread(self._launch))
                await asyncio.shield(self._launching)
                started = await asyncio.shield(self._started)
                if started.error:
                    raise WorkerLoadError(started.error)
                self.page_workers = started.page_workers
                await self._load(options)
        except TimeoutError:
            raise WorkerLoadError(f'The models did not load within {timeout} seconds') from None
        except WorkerGone as error:
            raise WorkerLoadError(f'{error} while loading the models') from error
        logger.info('Worker %s ready on %s', self._process.pid, started.device)

    async def reload(self, options: LayoutOptions, timeout: float | None) -> None:
        try:
            async with asyncio.timeout(timeout):
                await self._load(options)
        except TimeoutError:
            raise WorkerLoadError(f'The models did not load within {timeout} seconds') from None
        except WorkerGone as error:
            raise WorkerLoadError(f'{error} while loading the models') from error

    async def process(self, options: LayoutOptions, image: Path, file_name: str) -> Reply:
        return await self._request(lambda number: PageRequest(number, options, image, file_name))

    async def stop(self) -> None:
        self._stopping = True
        with suppress(OSError, ValueError):
            await asyncio.to_thread(self._send, None)
        await self._end(STOP_TIMEOUT)

    async def kill(self) -> None:
        self._stopping = True
        await self._end(0)

    def _launch(self) -> None:
        self._process.start()
        self._child_connection.close()
        threading.Thread(target=self._read, name='eynollah-worker-reader', daemon=True).start()

    async def _load(self, options: LayoutOptions) -> None:
        reply = await self._request(lambda number: LoadRequest(number, options))
        if reply.status != 'ok':
            raise WorkerLoadError(reply.message)

    async def _request(self, create: Callable[[int], object]) -> Reply:
        if self._gone:
            raise self._gone
        number = next(self._ids)
        future: asyncio.Future[Reply] = self._loop.create_future()
        future.add_done_callback(_retrieve_exception)
        self._pending[number] = future
        self._outstanding.add(number)
        try:
            with suppress(OSError, ValueError):
                await asyncio.to_thread(self._send, create(number))
            return await future
        finally:
            self._pending.pop(number, None)

    def _send(self, message: object) -> None:
        with self._send_lock:
            self._connection.send(message)

    def _read(self) -> None:
        try:
            while True:
                message = self._connection.recv()
                self._loop.call_soon_threadsafe(self._receive, message)
        except EOFError:
            pass
        except Exception as error:  # noqa: BLE001
            logger.debug('The worker connection failed: %s', error)
        self._process.join(STOP_TIMEOUT)
        if self._process.is_alive():
            self._process.kill()
        self._process.join()
        with suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._finish, self._process.exitcode)

    def _receive(self, message: object) -> None:
        if isinstance(message, Started):
            if not self._started.done():
                self._started.set_result(message)
            return
        if isinstance(message, Reply):
            self._outstanding.discard(message.id)
            future = self._pending.get(message.id)
            if future and not future.done():
                future.set_result(message)

    def _finish(self, exit_code: int | None) -> None:
        if self._stopping:
            self._gone = WorkerStopped('The worker was stopped')
        else:
            self._gone = WorkerCrashed(f'The worker {describe_exit(exit_code)}')
            logger.error('Worker %s %s', self._process.pid, describe_exit(exit_code))
        if not self._started.done():
            self._started.set_exception(self._gone)
        self._outstanding.clear()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(self._gone)
        self._connection.close()
        self._exited.set()

    async def _end(self, grace: float) -> None:
        if self._launching is None:
            return
        await asyncio.wait([self._launching])
        if self._process.pid is None:
            return
        if grace:
            with suppress(TimeoutError):
                async with asyncio.timeout(grace):
                    await self._exited.wait()
        if not self._exited.is_set():
            self._process.kill()
            await self._exited.wait()


class WorkerPool:
    def __init__(
        self,
        settings: WorkerSettings,
        *,
        loading_timeout: float | None,
        idle_timeout: float,
        target: Callable = worker,
    ):
        self._settings = settings
        self._loading_timeout = loading_timeout
        self._idle_timeout = idle_timeout
        self._target = target
        self._idle: dict[WorkerProcess, asyncio.TimerHandle] = {}
        self._workers: set[WorkerProcess] = set()
        self._tasks: set[asyncio.Task] = set()

    async def acquire(self, options: LayoutOptions) -> WorkerProcess:
        while self._idle:
            process, timer = self._idle.popitem()
            timer.cancel()
            if process.alive and await self._reload(process, options):
                return process
            self._workers.discard(process)
        return await self._start(options)

    async def release(self, process: WorkerProcess) -> None:
        if process.alive and not process.in_flight and self._idle_timeout:
            loop = asyncio.get_running_loop()
            self._idle[process] = loop.call_later(self._idle_timeout, self._expire, process)
            return
        await self.discard(process)

    async def discard(self, process: WorkerProcess) -> None:
        try:
            await asyncio.shield(self._spawn(process.kill()))
        finally:
            self._workers.discard(process)

    async def close(self) -> None:
        for timer in self._idle.values():
            timer.cancel()
        self._idle.clear()
        await asyncio.gather(*(process.kill() for process in self._workers), *self._tasks, return_exceptions=True)
        self._workers.clear()

    async def _start(self, options: LayoutOptions) -> WorkerProcess:
        process = WorkerProcess(self._settings, self._target)
        self._workers.add(process)
        try:
            await process.start(options, self._loading_timeout)
        except BaseException:
            await self.discard(process)
            raise
        return process

    async def _reload(self, process: WorkerProcess, options: LayoutOptions) -> bool:
        try:
            await process.reload(options, self._loading_timeout)
        except WorkerLoadError as error:
            await self.discard(process)
            if isinstance(error.__cause__, WorkerGone):
                return False
            raise
        except BaseException:
            await self.discard(process)
            raise
        return True

    def _expire(self, process: WorkerProcess) -> None:
        if self._idle.pop(process, None) is None:
            return
        logger.info('Releasing the idle worker')
        self._workers.discard(process)
        self._spawn(process.stop())

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task


def describe_exit(exit_code: int | None) -> str:
    if exit_code == -9:
        return 'was killed, most likely because the system ran out of memory'
    if exit_code is not None and exit_code < 0:
        return f'was terminated by signal {-exit_code}'
    return f'exited unexpectedly with code {exit_code}'


def _retrieve_exception(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()
