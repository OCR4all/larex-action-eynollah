# SPDX-License-Identifier: MIT
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from fake_worker import fake_worker

from larex_action.parameters import LayoutOptions
from larex_action.pool import WorkerCrashed, WorkerLoadError, WorkerPool, WorkerStopped, describe_exit
from larex_action.worker import WorkerSettings

OPTIONS = LayoutOptions()


@pytest.fixture
async def create_pool():
    pools = []

    def create(device='cpu', *, page_workers=1, loading_timeout=20, idle_timeout=30):
        settings = replace(WorkerSettings(Path('/models'), device=device), page_workers=page_workers)
        pools.append(
            WorkerPool(settings, loading_timeout=loading_timeout, idle_timeout=idle_timeout, target=fake_worker)
        )
        return pools[-1]

    yield create
    for pool in pools:
        await pool.close()


@pytest.fixture
def image(tmp_path):
    def create(behaviour):
        path = tmp_path / behaviour
        path.write_text(behaviour)
        return path

    return create


async def test_worker_processes_pages(create_pool, image):
    worker = await create_pool(page_workers=3).acquire(OPTIONS)
    assert worker.page_workers == 3
    replies = await asyncio.gather(*(worker.process(OPTIONS, image('slow'), 'page.png') for _ in range(3)))
    assert {reply.status for reply in replies} == {'ok'}
    assert len({reply.id for reply in replies}) == 3
    assert worker.in_flight == 0


async def test_released_worker_is_reused(create_pool):
    pool = create_pool()
    first = await pool.acquire(OPTIONS)
    await pool.release(first)
    assert await pool.acquire(OPTIONS) is first


async def test_dead_idle_worker_is_replaced(create_pool):
    pool = create_pool()
    first = await pool.acquire(OPTIONS)
    await pool.release(first)
    first._process.kill()
    second = await pool.acquire(OPTIONS)
    assert second is not first
    assert second.alive


async def test_idle_worker_expires(create_pool):
    pool = create_pool(idle_timeout=0.1)
    worker = await pool.acquire(OPTIONS)
    await pool.release(worker)
    await asyncio.wait_for(worker._exited.wait(), 10)
    assert worker._process.exitcode == 0


async def test_worker_is_stopped_without_an_idle_timeout(create_pool):
    pool = create_pool(idle_timeout=0)
    worker = await pool.acquire(OPTIONS)
    await pool.release(worker)
    assert not worker.alive


async def test_busy_worker_is_not_kept(create_pool, image):
    pool = create_pool()
    worker = await pool.acquire(OPTIONS)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(worker.process(OPTIONS, image('hang'), 'page.png'), 0.2)
    assert worker.in_flight == 1
    await pool.release(worker)
    assert not worker.alive


async def test_crash_is_reported_to_all_pages(create_pool, image):
    worker = await create_pool(page_workers=2).acquire(OPTIONS)
    results = await asyncio.gather(
        worker.process(OPTIONS, image('crash'), 'a.png'),
        worker.process(OPTIONS, image('hang'), 'b.png'),
        return_exceptions=True,
    )
    assert [type(result) for result in results] == [WorkerCrashed, WorkerCrashed]
    assert str(results[0]) == 'The worker exited unexpectedly with code 1'
    with pytest.raises(WorkerCrashed):
        await worker.process(OPTIONS, image('ok'), 'c.png')


async def test_kill_interrupts_pages(create_pool, image):
    pool = create_pool()
    worker = await pool.acquire(OPTIONS)
    page = asyncio.create_task(worker.process(OPTIONS, image('hang'), 'page.png'))
    await asyncio.sleep(0.1)
    await pool.discard(worker)
    with pytest.raises(WorkerStopped):
        await page


@pytest.mark.parametrize(
    ('device', 'message'),
    [
        ('unavailable', 'CUDA is unavailable'),
        ('broken', 'The models are damaged'),
        ('crash', 'The worker exited unexpectedly with code 7 while loading the models'),
    ],
)
async def test_load_failures(create_pool, device, message):
    with pytest.raises(WorkerLoadError, match=message):
        await create_pool(device).acquire(OPTIONS)


async def test_loading_timeout(create_pool):
    pool = create_pool('hang', loading_timeout=2)
    with pytest.raises(WorkerLoadError, match='did not load within 2 seconds'):
        await pool.acquire(OPTIONS)
    assert not pool._workers


async def test_cancelled_start_leaves_no_process(create_pool):
    pool = create_pool('hang')
    start = asyncio.create_task(pool.acquire(OPTIONS))
    await asyncio.sleep(0.01)
    worker = next(iter(pool._workers))
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start
    assert not worker._process.is_alive()


async def test_close_stops_all_workers(create_pool):
    pool = create_pool()
    workers = [await pool.acquire(OPTIONS) for _ in range(2)]
    await pool.release(workers[0])
    await pool.close()
    assert not any(worker._process.is_alive() for worker in workers)


def test_exit_descriptions():
    assert 'out of memory' in describe_exit(-9)
    assert describe_exit(-11) == 'was terminated by signal 11'
