# SPDX-License-Identifier: MIT
import asyncio
from dataclasses import replace

import pytest
from conftest import FakeContext
from fake_worker import fake_worker
from larex_actions import ActionCancelled

from larex_action import runner as runner_module
from larex_action.app import worker_settings
from larex_action.models import MODEL_FILES
from larex_action.pool import WorkerPool
from larex_action.runner import Runner, result_file_name


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(runner_module, 'CANCEL_POLL_SECONDS', 0.05)
    monkeypatch.setattr(runner_module, 'QUEUE_POLL_SECONDS', 0.05)


@pytest.fixture
async def create_runner(settings):
    runners = []

    def create(page_workers=1, device='cpu', **overrides):
        current = settings(**overrides)
        pool = WorkerPool(
            replace(worker_settings(current), page_workers=page_workers, device=device),
            loading_timeout=20,
            idle_timeout=30,
            target=fake_worker,
        )
        runners.append(Runner(current, pool))
        return runners[-1]

    yield create
    for runner in runners:
        await runner.close()


def statuses(ctx):
    return [ctx.outcomes[page.id] for page in ctx.input.pages if page.id in ctx.outcomes]


def worker_ids(ctx):
    return {content for contents in ctx.contents_by_page.values() for content in contents}


@pytest.mark.parametrize('page_workers', [1, 3])
async def test_run(create_runner, page_workers):
    ctx = FakeContext(['ok', 'empty', 'failed', 'slow', 'unavailable', None, 'ok'])
    await create_runner(page_workers)(ctx)
    assert statuses(ctx) == ['SUCCEEDED', 'SKIPPED', 'FAILED', 'SUCCEEDED', 'FAILED', 'FAILED', 'SUCCEEDED']
    assert ctx.reasons['page-0'] is None
    assert ctx.reasons['page-1'] == 'No regions were found, existing annotations were kept'
    assert ctx.reasons['page-2'] == 'The image format is not supported'
    assert ctx.reasons['page-4'] == 'The image could not be downloaded: ConnectError'
    assert ctx.reasons['page-5'] == 'The page has no image'
    assert [file.file_name for file in ctx.files['page-0']] == ['page-0.xml']
    assert not ctx.files['page-1']
    assert ctx.completed == 'Segmented 3 of 7 page(s), 3 failed, 1 without any regions'
    assert ctx.failed is None
    assert ctx.heartbeats[-1][0] == 100


async def test_consecutive_runs_reuse_the_worker(create_runner):
    runner = create_runner()
    contexts = [FakeContext(['ok']), FakeContext(['ok'])]
    for ctx in contexts:
        await runner(ctx)
    assert worker_ids(contexts[0]) == worker_ids(contexts[1])


async def test_empty_input_fails_the_run(create_runner):
    ctx = FakeContext([])
    await create_runner()(ctx)
    assert ctx.failed == 'The action input contains no pages'
    assert ctx.completed is None


async def test_invalid_parameters_fail_the_run(create_runner):
    ctx = FakeContext(['ok'], {'minColumns': 9})
    await create_runner()(ctx)
    assert ctx.failed.startswith('Invalid parameters: minColumns')


async def test_missing_models_fail_the_run(create_runner, model_directory):
    (model_directory / MODEL_FILES['table'].path).unlink()
    ctx = FakeContext(['ok'], {'tables': True})
    await create_runner()(ctx)
    assert ctx.failed == 'The processor is missing the models: table'
    ctx = FakeContext(['ok'])
    await create_runner()(ctx)
    assert statuses(ctx) == ['SUCCEEDED']


async def test_load_error_fails_the_run(create_runner):
    ctx = FakeContext(['ok'])
    await create_runner(device='broken')(ctx)
    assert ctx.failed == 'The models are damaged'
    assert not ctx.outcomes


async def test_timeout_fails_the_page(create_runner):
    ctx = FakeContext(['ok', 'hang', 'ok'])
    await create_runner(ACTION_PROCESSING_TIMEOUT=1)(ctx)
    assert statuses(ctx) == ['SUCCEEDED', 'FAILED', 'SUCCEEDED']
    assert ctx.reasons['page-1'] == 'The page was not finished within 1 seconds'
    assert len(worker_ids(ctx)) == 2


async def test_timeout_restarts_other_pages(create_runner):
    ctx = FakeContext(['hang', *['slow'] * 8])
    await create_runner(2, ACTION_PROCESSING_TIMEOUT=1)(ctx)
    assert statuses(ctx) == ['FAILED', *['SUCCEEDED'] * 8]


async def test_timeout_fails_the_run(create_runner):
    ctx = FakeContext(['ok', 'hang', 'ok'])
    await create_runner(ACTION_PROCESSING_TIMEOUT=1, ACTION_TIMEOUT_MODE='fail_action')(ctx)
    assert ctx.failed == 'The page was not finished within 1 seconds'
    assert ctx.completed is None


@pytest.mark.parametrize('behaviour', ['crash', 'broken'])
@pytest.mark.parametrize('page_workers', [1, 3])
async def test_crash_only_fails_the_responsible_page(create_runner, behaviour, page_workers):
    ctx = FakeContext(['slow', behaviour, 'slow', 'slow', 'ok'])
    await create_runner(page_workers)(ctx)
    assert statuses(ctx) == ['SUCCEEDED', 'FAILED', 'SUCCEEDED', 'SUCCEEDED', 'SUCCEEDED']
    assert ctx.completed == 'Segmented 4 of 5 page(s), 1 failed'


@pytest.mark.parametrize('page_workers', [1, 2])
async def test_repeated_crashes_fail_the_run(create_runner, page_workers):
    ctx = FakeContext(['crash'] * 6)
    await create_runner(page_workers)(ctx)
    assert ctx.failed.startswith('Stopped after 3 consecutive worker failures')
    assert len(ctx.outcomes) <= 2
    assert ctx.completed is None


async def test_simultaneous_crashes_do_not_fail_the_run(create_runner):
    ctx = FakeContext(['crash', 'crash', 'ok'])
    await create_runner(3)(ctx)
    assert statuses(ctx) == ['FAILED', 'FAILED', 'SUCCEEDED']


async def test_rejected_result_is_counted_as_failed(create_runner):
    ctx = FakeContext(['ok', 'ok'])
    ctx.reject.add('page-0')
    await create_runner()(ctx)
    assert statuses(ctx) == ['SUCCEEDED']
    assert ctx.completed == 'Segmented 1 of 2 page(s), 1 failed'


async def test_cancellation_interrupts_a_page(create_runner):
    runner = create_runner(2)
    ctx = FakeContext(['hang', 'hang', 'ok'])
    run = asyncio.create_task(runner(ctx))
    while ctx.downloads < 3:
        await asyncio.sleep(0.01)
    ctx.cancel_requested = True
    with pytest.raises(ActionCancelled):
        await asyncio.wait_for(run, 5)
    assert ctx.completed is None
    assert ctx.failed is None
    assert not runner._pool._workers


async def test_cancellation_while_waiting(create_runner):
    runner = create_runner()
    first = asyncio.create_task(runner(FakeContext(['hang'])))
    await asyncio.sleep(0.05)
    waiting = FakeContext(['ok'])
    second = asyncio.create_task(runner(waiting))
    while not waiting.heartbeats:
        await asyncio.sleep(0.01)
    assert waiting.heartbeats[0] == (None, 'Waiting for another run to finish')
    waiting.cancel_requested = True
    with pytest.raises(ActionCancelled):
        await asyncio.wait_for(second, 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first


async def test_run_waits_for_a_free_slot(create_runner):
    runner = create_runner()
    contexts = [FakeContext(['slow', 'slow']), FakeContext(['ok'])]
    await asyncio.gather(*(runner(ctx) for ctx in contexts))
    assert all(ctx.completed for ctx in contexts)
    assert worker_ids(contexts[0]) == worker_ids(contexts[1])


async def test_waiting_runs_start_in_order(create_runner):
    runner = create_runner()
    contexts = [FakeContext(['slow', 'slow']), FakeContext(['ok']), FakeContext(['ok'])]
    finished = []

    async def run(index):
        await runner(contexts[index])
        finished.append(index)

    await asyncio.gather(*(run(index) for index in range(3)))
    assert finished == [0, 1, 2]
    assert not runner._slots.locked()


async def test_shutdown_reports_active_runs(create_runner):
    runner = create_runner()
    ctx = FakeContext(['hang'])
    run = asyncio.create_task(runner(ctx))
    while not ctx.downloads:
        await asyncio.sleep(0.01)
    await runner.close()
    assert run.cancelled()
    assert ctx.failed == 'The processor is shutting down'


async def test_temporary_images_are_removed(create_runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner_module.tempfile, 'tempdir', str(tmp_path))
    await create_runner(2)(FakeContext(['ok', 'failed', 'empty']))
    assert not list(tmp_path.glob('eynollah-*'))


@pytest.mark.parametrize(
    ('name', 'expected'), [('0001.png', '0001.xml'), ('Blatt 3 (verso)', 'Blatt-3-verso.xml'), ('---', 'page.xml')]
)
def test_result_file_name(name, expected):
    page = FakeContext(['ok']).input.pages[0].model_copy(update={'name': name})
    assert result_file_name(page) == expected
