from fastapi import APIRouter

from app.api.v1.endpoints import health
from app.api.v1.endpoints import users
from app.api.v1.endpoints import auth
from app.api.v1.endpoints import gee2

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(gee2.router, prefix="/gee2", tags=["gee2"])

# This is the main API router that includes all endpoint routers