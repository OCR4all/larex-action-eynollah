# SPDX-License-Identifier: MIT
import hashlib
import io
import zipfile

import httpx
import pytest

from larex_action import models
from larex_action.models import ModelFile, main, missing_models, required_models
from larex_action.parameters import LayoutOptions

CONTENT = {'page': b'page model', 'table': b'table model'}


@pytest.fixture
def archive(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as target:
        for name, content in CONTENT.items():
            target.writestr(f'models_eynollah/{name}.onnx', content)
        target.writestr('models_eynollah/unused.onnx', b'unused')
    data = buffer.getvalue()
    files = {
        name: ModelFile(f'models_eynollah/{name}.onnx', len(content), hashlib.sha256(content).hexdigest())
        for name, content in CONTENT.items()
    }
    monkeypatch.setattr(models, 'MODEL_FILES', files)
    monkeypatch.setattr(models, 'ARCHIVE_SIZE', len(data))
    monkeypatch.setattr(models, 'ARCHIVE_SHA256', hashlib.sha256(data).hexdigest())
    return data


@pytest.fixture
def serve(monkeypatch):
    def install(*responses):
        remaining = list(responses)
        client = httpx.Client(transport=httpx.MockTransport(lambda _: remaining.pop(0)))
        monkeypatch.setattr(models.httpx, 'stream', client.stream)
        return remaining

    return install


def download(directory):
    return main(['download', '--directory', str(directory)])


def test_download_installs_the_required_files(archive, serve, tmp_path):
    serve(httpx.Response(200, content=archive))
    assert download(tmp_path / 'models') == 0
    installed = sorted(path.name for path in (tmp_path / 'models').rglob('*') if path.is_file())
    assert installed == ['page.onnx', 'table.onnx']
    assert (tmp_path / 'models/models_eynollah/page.onnx').read_bytes() == CONTENT['page']


def test_download_keeps_complete_models(archive, serve, tmp_path):
    serve(httpx.Response(200, content=archive))
    assert download(tmp_path) == 0
    serve()
    assert download(tmp_path) == 0


def test_download_is_repeated_after_a_failure(archive, serve, tmp_path):
    remaining = serve(
        httpx.Response(503), httpx.Response(200, content=b'truncated'), httpx.Response(200, content=archive)
    )
    assert download(tmp_path) == 0
    assert not remaining


def test_download_rejects_a_changed_archive(archive, serve, tmp_path, capsys):
    serve(*(httpx.Response(200, content=b'changed') for _ in range(3)))
    assert download(tmp_path) == 1
    assert 'does not match the pinned checksum' in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_verify_reports_damaged_and_missing_files(archive, serve, tmp_path, capsys):
    serve(httpx.Response(200, content=archive))
    download(tmp_path)
    (tmp_path / 'models_eynollah/page.onnx').write_bytes(b'damaged')
    (tmp_path / 'models_eynollah/table.onnx').unlink()
    assert main(['verify', '--directory', str(tmp_path)]) == 1
    errors = capsys.readouterr().err
    assert 'page:' in errors and 'does not match the published checksum' in errors
    assert 'table:' in errors and 'is missing' in errors


def test_required_models():
    assert required_models(LayoutOptions(lines_only=True, full_layout=False)) == [
        'col_classifier',
        'page',
        'textline',
        'region_1_2',
    ]
    everything = LayoutOptions(binarize=True, model_reading_order=True, tables=True)
    assert set(required_models(everything)) == set(models.MODEL_FILES)


def test_missing_models(model_directory):
    (model_directory / models.MODEL_FILES['table'].path).unlink()
    assert missing_models(model_directory, ['page', 'table']) == ['table']


def test_manifest_matches_the_eynollah_release():
    from eynollah.model_zoo.default_specs import DEFAULT_MODEL_SPECS

    for name, file in models.MODEL_FILES.items():
        specified = DEFAULT_MODEL_SPECS.get(name, '').filename
        assert specified.rsplit('.', 1)[0] == file.path.rsplit('.', 1)[0]
