"""FastAPI application factory and process entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.config import API_DESCRIPTION, API_TITLE, API_VERSION
from api.routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create a documented FastAPI application with safe unhandled errors."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        contact={"name": "Project maintainer"},
    )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, error: Exception) -> JSONResponse:
        """Log internal details while returning a stable public response."""
        logger.exception("Unhandled API error", exc_info=error)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    app.include_router(router)
    return app


app = create_app()
