# SPDX-License-Identifier: MIT
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
from PIL import Image

from larex_action import worker
from larex_action.images import UnreadableImage
from larex_action.parameters import LayoutOptions
from larex_action.worker import LoadRequest, PageRequest, Processor, WorkerSettings, classify, default_page_workers

OPTIONS = LayoutOptions()


@pytest.fixture
def replies():
    return Queue()


@pytest.fixture
def processor(replies):
    settings = WorkerSettings(Path('/models'), device='cpu', page_workers=2, max_image_pixels=100)
    return Processor(SimpleNamespace(send=replies.put), settings)


@pytest.fixture
def engine(processor, monkeypatch):
    engine = SimpleNamespace(load=lambda options: None, process=lambda image, file_name, options: b'<PcGts/>')
    monkeypatch.setattr(processor, '_create_engine', lambda: engine)
    return engine


@pytest.fixture
def image(tmp_path):
    path = tmp_path / 'page.png'
    Image.new('L', (10, 10)).save(path)
    return path


def answer(processor, replies, request):
    processor.submit(request)
    return replies.get(timeout=5)


def raiser(error):
    def call(*_):
        raise error

    return call


def test_load(processor, engine, replies):
    assert answer(processor, replies, LoadRequest(1, OPTIONS)).status == 'ok'


def test_load_error(processor, engine, replies):
    engine.load = raiser(RuntimeError('damaged  model'))
    reply = answer(processor, replies, LoadRequest(1, OPTIONS))
    assert (reply.status, reply.message) == ('failed', 'RuntimeError: damaged model')


def test_page(processor, engine, replies, image):
    reply = answer(processor, replies, PageRequest(3, OPTIONS, image, 'page.png'))
    assert (reply.id, reply.status, reply.xml) == (3, 'ok', b'<PcGts/>')
    assert reply.seconds > 0


def test_empty_page(processor, engine, replies, image):
    engine.process = lambda *_: None
    assert answer(processor, replies, PageRequest(1, OPTIONS, image, 'page.png')).status == 'empty'


def test_page_error(processor, engine, replies, image):
    engine.process = raiser(ZeroDivisionError())
    reply = answer(processor, replies, PageRequest(1, OPTIONS, image, 'page.png'))
    assert (reply.status, reply.message) == ('failed', 'ZeroDivisionError')


def test_too_large_image(processor, engine, replies, tmp_path):
    path = tmp_path / 'page.png'
    Image.new('L', (11, 10)).save(path)
    reply = answer(processor, replies, PageRequest(1, OPTIONS, path, 'page.png'))
    assert reply.status == 'failed'
    assert reply.message == 'The image has 11x10 pixels, more than the limit of 100'


def test_classification():
    runtime_error = type('RuntimeException', (Exception,), {'__module__': 'onnxruntime.capi.state'})('out of memory')
    wrapped = ValueError('prediction failed')
    wrapped.__cause__ = runtime_error
    assert classify(runtime_error) == 'broken'
    assert classify(wrapped) == 'broken'
    assert classify(MemoryError()) == 'broken'
    assert classify(UnreadableImage('unsupported')) == 'failed'


@pytest.mark.parametrize(
    ('device', 'threads', 'expected'), [('cpu', 16, 1), ('cuda', 2, 2), ('cuda', 16, 4), ('cuda:1', 16, 4)]
)
def test_default_page_workers(device, threads, expected):
    assert default_page_workers(device, threads) == expected


def test_unavailable_device(replies, monkeypatch):
    monkeypatch.setattr('onnxruntime.get_available_providers', lambda: ['CPUExecutionProvider'])
    connection = SimpleNamespace(send=replies.put)
    assert worker._start(connection, WorkerSettings(Path('/models'), device='cuda')) is None
    assert 'only contains the CPU runtime' in replies.get(timeout=1).error


def test_auto_device_falls_back_to_the_cpu(replies, monkeypatch):
    monkeypatch.setattr('onnxruntime.get_available_providers', lambda: ['CPUExecutionProvider'])
    processor = worker._start(SimpleNamespace(send=replies.put), WorkerSettings(Path('/models')))
    started = replies.get(timeout=1)
    assert processor
    assert (started.device, started.page_workers) == ('cpu', 1)
