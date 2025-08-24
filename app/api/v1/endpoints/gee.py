import os
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

import ee
from google.oauth2 import service_account

from app.db.session import get_db
from app.utils.common import log_error
from app.utils.response import create_response

router = APIRouter()
logger = logging.getLogger(__name__)

# Paths and env variables
# KEY_PATH = os.getenv("GEE_KEY_PATH", "/app/key.json")
KEY_PATH = os.getenv("GEE_KEY_PATH", "D:\\Projects\\Projects\\58-AGRAS\\key.json")
PROJECT_ID = os.getenv("EARTHENGINE_PROJECT", 'tribal-datum-463421-h3')

# Load service account info
try:
    with open(KEY_PATH, "r") as key_file:
        service_info = json.load(key_file)
    logger.info("Loaded Earth Engine service account from %s", KEY_PATH)
except FileNotFoundError:
    logger.error("GEE key file not found at %s", KEY_PATH)
    raise

# Determine project ID
if not PROJECT_ID:
    PROJECT_ID = service_info.get("project_id")
    if not PROJECT_ID:
        raise RuntimeError(
            "Missing Earth Engine project ID. Set EARTHENGINE_PROJECT or include project_id in key.json"
        )

# Build credentials and initialize EE
credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/earthengine"],
    quota_project_id=PROJECT_ID,
)

_gee_error = None
try:
    ee.Initialize(credentials=credentials, project=PROJECT_ID)
    logger.info("Earth Engine initialized for project %s", PROJECT_ID)
except Exception as init_err:
    _gee_error = init_err
    logger.exception("Failed to initialize Earth Engine: %s", init_err)

@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="GEE + Health check endpoint",
    description="Verifica que el servicio, la DB y GEE estén disponibles",
)
async def gee_health_check(db: AsyncSession = Depends(get_db)):
    """
    Health check for the application, database and Google Earth Engine.
    """
    log_error("Health check with GEE performed")

    # # Database check
    # try:
    #     await db.execute(text("SELECT 1"))
    #     db_status = "connected"
    # except Exception as db_err:
    #     db_status = f"error: {db_err}"
    #     logger.error("Database health check failed: %s", db_err)

    # Earth Engine check
    if _gee_error:
        gee_status = f"init-error: {_gee_error}"
    else:
        try:
            # Fetch a test number from Earth Engine
            test_num = ee.Number(1).getInfo()
            print('test_nume', test_num)
            gee_status = "connected"
            logger.info("GEE test number: %s", test_num)
        except Exception as gee_err:
            gee_status = f"error: {gee_err}"
            logger.error("Earth Engine health check failed: %s", gee_err)
            test_num = None

    # Build response payload
    payload = {
        "status": "ok",
        "message": "Service + GEE health check",
        # "database": db_status,
        "earth_engine": gee_status,
        "test_number": test_num,
        "version": "0.1.0",
    }

    return create_response(data=payload, message="Health check successful")
