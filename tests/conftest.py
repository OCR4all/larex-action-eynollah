# SPDX-License-Identifier: MIT
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest
from larex_actions import ActionCancelled, PageResultRejected, ResultBuilder
from larex_actions.models import ActionInput

from larex_action.config import Settings
from larex_action.models import MODEL_FILES


class FakeContext:
    run_id = 'run-1'

    def __init__(self, behaviours, parameters=None):
        self.contents = dict(enumerate(behaviours))
        self.input = ActionInput.model_validate(
            {
                'protocolVersion': 1,
                'runId': self.run_id,
                'processorKey': 'eynollah-segmentation',
                'projectId': 'project-1',
                'parameters': parameters or {},
                'pages': [page(index, content) for index, content in self.contents.items()],
            }
        )
        self.outcomes = {}
        self.reasons = {}
        self.files = {}
        self.contents_by_page = {}
        self.heartbeats = []
        self.completed = None
        self.failed = None
        self.cancel_requested = False
        self.reject = set()
        self.downloads = 0

    async def pull_input(self):
        return self.input

    async def heartbeat(self, progress_percent=None, status_message=None, *, raise_on_cancel=False):
        self.heartbeats.append((progress_percent, status_message))
        if raise_on_cancel:
            await self.check_cancelled()

    async def check_cancelled(self):
        if self.cancel_requested:
            raise ActionCancelled('LAREX requested cancellation')

    async def download_to_path(self, file, path):
        self.downloads += 1
        content = self.contents[int(file.id)]
        if content == 'unavailable':
            raise httpx.ConnectError('connection refused')
        Path(path).write_text(content)
        return path

    def result_builder(self):
        return ResultBuilder()

    async def submit_page_results(self, page_id, results, message, *, page_status, reason):
        if page_id in self.reject:
            request = httpx.Request('POST', 'https://larex.example.org')
            raise PageResultRejected('invalid PAGE XML', request=request, response=httpx.Response(422, request=request))
        assert page_id not in self.outcomes
        self.outcomes[page_id] = page_status
        self.reasons[page_id] = reason
        self.files[page_id] = results.files
        with ExitStack() as stack:
            self.contents_by_page[page_id] = [file[1] for _, file in results.httpx_files(exit_stack=stack)[1:]]

    async def complete(self, message=None):
        self.completed = message

    async def fail(self, message):
        self.failed = message


def page(index, content):
    file = {'id': str(index), 'fileName': f'page {index}.png', 'downloadUrl': f'https://larex.example.org/{index}'}
    return {'id': f'page-{index}', 'name': f'page {index}', 'images': [] if content is None else [file]}


@pytest.fixture
def model_directory(tmp_path):
    directory = tmp_path / 'models'
    for file in MODEL_FILES.values():
        path = directory / file.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return directory


@pytest.fixture
def settings(model_directory):
    def create(**overrides):
        values = {
            'LAREX_PROCESSOR_ID': 'eynollah-segmentation',
            'LAREX_DISPATCH_HMAC_SECRET': 'test-dispatch-secret',
            'LAREX_ALLOWED_CALLBACK_ORIGINS': ['https://larex.example.org'],
            'EYNOLLAH_MODEL_DIR': model_directory,
        }
        return Settings(**(values | overrides))

    return create
