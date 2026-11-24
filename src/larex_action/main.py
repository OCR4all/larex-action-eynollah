# SPDX-License-Identifier: MIT
from .app import configure_logging, create_app
from .config import Settings

settings = Settings()
configure_logging(settings)
app = create_app(settings)
