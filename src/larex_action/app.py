# SPDX-License-Identifier: MIT
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from larex_actions.fastapi import create_larex_action_app

from .config import Settings
from .pool import WorkerPool
from .resources import available_cpus
from .runner import Runner
from .worker import WorkerSettings

logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(level=log_level(settings), format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    if settings.LAREX_SDK_TRANSPORT_DEBUG:
        logging.getLogger('larex_actions.transport').setLevel(logging.DEBUG)


def log_level(settings: Settings) -> int:
    return logging.DEBUG if settings.ACTION_DEBUG else logging.INFO


def worker_settings(settings: Settings) -> WorkerSettings:
    cpus = max(1, available_cpus() // settings.ACTION_MAX_CONCURRENT_RUNS)
    return WorkerSettings(
        model_directory=settings.EYNOLLAH_MODEL_DIR,
        device=settings.EYNOLLAH_DEVICE,
        threads=settings.EYNOLLAH_CPU_THREADS or cpus,
        page_workers=settings.EYNOLLAH_PAGE_WORKERS,
        memory_limits=settings.EYNOLLAH_GPU_MEMORY_LIMITS,
        max_image_pixels=settings.EYNOLLAH_MAX_IMAGE_PIXELS,
        log_level=log_level(settings),
    )


def create_app(settings: Settings) -> FastAPI:
    pool = WorkerPool(
        worker_settings(settings),
        loading_timeout=settings.ACTION_LOADING_TIMEOUT,
        idle_timeout=settings.ACTION_IDLE_TIMEOUT,
    )
    runner = Runner(settings, pool)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await runner.close()

    return create_larex_action_app(
        app=FastAPI(title='LAREX Action: eynollah', lifespan=lifespan, openapi_url=None),
        processor_id=settings.LAREX_PROCESSOR_ID,
        dispatch_secret=settings.LAREX_DISPATCH_HMAC_SECRET,
        handler=runner,
        allowed_callback_origins=settings.LAREX_ALLOWED_CALLBACK_ORIGINS,
        route_prefixes=settings.LAREX_ROUTE_PREFIXES,
    )
