# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_ignore_empty=True, env_parse_none_str='None')

    LAREX_PROCESSOR_ID: str
    LAREX_DISPATCH_HMAC_SECRET: str
    LAREX_ALLOWED_CALLBACK_ORIGINS: list[str]
    LAREX_ROUTE_PREFIXES: list[str] = []
    LAREX_SDK_TRANSPORT_DEBUG: bool = False

    ACTION_DEBUG: bool = False
    ACTION_MAX_CONCURRENT_RUNS: PositiveInt = 1
    ACTION_LOADING_TIMEOUT: PositiveInt | None = 300
    ACTION_PROCESSING_TIMEOUT: PositiveInt | None = 900
    ACTION_TIMEOUT_MODE: Literal['fail_action', 'fail_page'] = 'fail_page'
    ACTION_IDLE_TIMEOUT: NonNegativeInt = 600

    EYNOLLAH_MODEL_DIR: Path = Path('/models')
    EYNOLLAH_DEVICE: str = Field(default='auto', pattern=r'^(auto|cpu|cuda(:\d+)?)$')
    EYNOLLAH_PAGE_WORKERS: PositiveInt | None = None
    EYNOLLAH_CPU_THREADS: PositiveInt | None = None
    EYNOLLAH_GPU_MEMORY_LIMITS: bool = False
    EYNOLLAH_MAX_IMAGE_PIXELS: PositiveInt = 300_000_000
