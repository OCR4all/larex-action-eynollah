# SPDX-License-Identifier: MIT
import logging
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

from .parameters import LayoutOptions

MAX_DEFAULT_PAGE_WORKERS = 4

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerSettings:
    model_directory: Path
    device: str = 'auto'
    threads: int = 1
    page_workers: int | None = None
    memory_limits: bool = False
    max_image_pixels: int = 300_000_000
    log_level: int = logging.INFO


@dataclass(frozen=True)
class LoadRequest:
    id: int
    options: LayoutOptions


@dataclass(frozen=True)
class PageRequest:
    id: int
    options: LayoutOptions
    image: Path
    file_name: str


@dataclass(frozen=True)
class Started:
    device: str = ''
    page_workers: int = 1
    error: str = ''


@dataclass(frozen=True)
class Reply:
    id: int
    status: Literal['ok', 'empty', 'failed', 'broken']
    message: str = ''
    xml: bytes | None = field(default=None, repr=False)
    seconds: float = 0


def worker(connection: Connection, settings: WorkerSettings) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=settings.log_level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger('eynollah').setLevel(logging.DEBUG if settings.log_level <= logging.DEBUG else logging.ERROR)
    try:
        processor = _start(connection, settings)
        while processor and (request := connection.recv()) is not None:
            processor.submit(request)
    except (EOFError, OSError):
        pass
    os._exit(0)


def _start(connection: Connection, settings: WorkerSettings) -> 'Processor | None':
    from .engine import resolve_device

    try:
        device = resolve_device(settings.device)
    except Exception as error:  # noqa: BLE001
        connection.send(Started(error=str(error)))
        return None
    settings = replace(
        settings,
        device=device,
        page_workers=settings.page_workers or default_page_workers(device, settings.threads),
    )
    connection.send(Started(device=device, page_workers=settings.page_workers))
    return Processor(connection, settings)


def default_page_workers(device: str, threads: int) -> int:
    return 1 if device == 'cpu' else min(MAX_DEFAULT_PAGE_WORKERS, threads)


class Processor:
    def __init__(self, connection: Connection, settings: WorkerSettings):
        self._connection = connection
        self._settings = settings
        self._send_lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._engine = None
        self._executor = ThreadPoolExecutor(settings.page_workers, thread_name_prefix='page')

    def submit(self, request: LoadRequest | PageRequest) -> None:
        handler = self._load if isinstance(request, LoadRequest) else self._process
        self._executor.submit(self._answer, handler, request)

    def _answer(self, handler, request) -> None:
        started = time.perf_counter()
        try:
            status, xml, message = handler(request)
        except Exception as error:  # noqa: BLE001
            logger.exception('Request %s failed', request.id)
            status, xml, message = classify(error), None, describe(error)
        reply = Reply(request.id, status, message, xml, time.perf_counter() - started)
        try:
            with self._send_lock:
                self._connection.send(reply)
        except OSError:
            os._exit(0)

    def _load(self, request: LoadRequest):
        self._get_engine().load(request.options)
        return 'ok', None, ''

    def _process(self, request: PageRequest):
        from .images import load_rgb

        image = load_rgb(request.image, self._settings.max_image_pixels)
        xml = self._get_engine().process(image, request.file_name, request.options)
        return ('ok' if xml else 'empty'), xml, ''

    def _get_engine(self):
        with self._engine_lock:
            if self._engine is None:
                self._engine = self._create_engine()
            return self._engine

    def _create_engine(self):
        import cv2

        from .engine import DeviceSettings, LayoutEngine

        settings = self._settings
        cv2.setNumThreads(max(1, settings.threads // settings.page_workers))
        return LayoutEngine(
            settings.model_directory,
            DeviceSettings(device=settings.device, threads=settings.threads, memory_limits=settings.memory_limits),
        )


def classify(error: BaseException) -> Literal['failed', 'broken']:
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, MemoryError) or type(current).__module__.startswith('onnxruntime'):
            return 'broken'
        current = current.__cause__ or current.__context__
    return 'failed'


def describe(error: BaseException) -> str:
    text = ' '.join(str(error).split())[:500]
    if isinstance(error, ValueError) and type(error).__module__.startswith('larex_action'):
        return text
    return f'{type(error).__name__}: {text}' if text else type(error).__name__
