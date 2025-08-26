# app/main.py
from fastapi import FastAPI
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.v1.api import api_router
from app.exceptions.handlers import add_exception_handlers
from app.middlewares.setup import setup_middlewares
from app.utils.docs import setup_swagger_documentation
from app.utils.response import create_response

# 👇 importa el inicializador
from app.services.gee_service import ensure_ee_initialized

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up the application...")

    # 👇 inicializa Earth Engine una sola vez al arranque
    try:
        ensure_ee_initialized()
        logger.info("Earth Engine initialized (startup)")
    except Exception as e:
        # No bloquees el arranque si falla; loggea y deja que las rutas den error controlado
        logger.warning("EE init failed/deferred: %s", e)

    yield
    logger.info("Shutting down the application...")

app = FastAPI(
    title="FastAPI Application",
    description="FastAPI application with SQLAlchemy and PostgreSQL",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

setup_swagger_documentation(app, settings.API_V1_STR)
add_exception_handlers(app)
setup_middlewares(app)

app.include_router(prefix=settings.API_V1_STR, router=api_router)

@app.get("/", tags=["health"])
def root():
    data = {"status": "ok", "message": "API is running"}
    return create_response(data=data, message="API is running")
