# app/core/config.py
from __future__ import annotations
import base64
import json
import os
import platform
import secrets
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv(override=True)

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "t", "yes", "y")

class Settings:
    # ---- Entorno ----
    APP_ENV: str
    IS_DOCKER: bool
    IS_WINDOWS: bool

    # ---- API / Auth ----
    API_V1_STR: str
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    ALGORITHM: str
    JWT_AUDIENCE: str
    JWT_ISSUER: str
    PUBLIC_BASE_URL: Optional[str]

    # ---- CORS ----
    BACKEND_CORS_ORIGINS: List[str]

    # ---- DB ----
    POSTGRES_SERVER: str
    POSTGRES_PORT: Optional[str]
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    DATABASE_URL: Optional[str]

    # ---- GEE ----
    EARTHENGINE_PROJECT: Optional[str]
    GEE_KEY_PATH: Optional[str]
    GEE_KEY_JSON: Optional[str]
    GEE_KEY_B64: Optional[str]
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str]
    TILES_DEFAULT_CID: str
    S2_COLLECTION: str
    TILE_CACHE_SECONDS: int
    TILE_STALE_SECONDS: int

    # ---- Otros ----
    CLIENT_IDS: str

    def __init__(self) -> None:
        # -------- Base de entorno --------
        self.APP_ENV = os.getenv("APP_ENV", os.getenv("ENVIRONMENT", "local")).lower()
        self.IS_DOCKER = _bool_env("DOCKERIZED", os.path.exists("/.dockerenv"))
        self.IS_WINDOWS = platform.system().lower() == "windows"

        # -------- API / Auth --------
        self.API_V1_STR = os.getenv("API_V1_STR", "/api/v1")
        self.SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 8)))
        self.ALGORITHM = os.getenv("ALGORITHM", "HS256")
        self.JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "*")
        self.PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
        self.JWT_ISSUER = os.getenv("JWT_ISSUER", (self.PUBLIC_BASE_URL or "http://localhost:8000"))

        # -------- CORS --------
        cors_origins = os.getenv("BACKEND_CORS_ORIGINS", "*")
        if cors_origins == "*":
            self.BACKEND_CORS_ORIGINS = ["*"]
        elif cors_origins.startswith("[") and cors_origins.endswith("]"):
            self.BACKEND_CORS_ORIGINS = json.loads(cors_origins)
        else:
            self.BACKEND_CORS_ORIGINS = [i.strip() for i in cors_origins.split(",")]

        # -------- DB --------
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")
        self.POSTGRES_PORT = os.getenv("POSTGRES_PORT")  # opcional
        self.POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
        self.POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
        self.POSTGRES_DB = os.getenv("POSTGRES_DB", "app")

        if (self.IS_DOCKER or self.APP_ENV in ("docker", "production")) and \
           self.POSTGRES_SERVER in ("localhost", "127.0.0.1"):
            self.POSTGRES_SERVER = os.getenv("POSTGRES_HOST", "db")

        # -------- GEE --------
        self.EARTHENGINE_PROJECT = os.getenv("EARTHENGINE_PROJECT")
        self.GEE_KEY_JSON = os.getenv("GEE_KEY_JSON")
        self.GEE_KEY_B64 = os.getenv("GEE_KEY_B64")
        self.GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

        default_local_path = r"D:\Projects\Projects\58-AGRAS\key.json" if self.IS_WINDOWS else "/app/key.json"
        default_docker_path = "/app/key.json"
        self.GEE_KEY_PATH = os.getenv("GEE_KEY_PATH") or (
            default_docker_path if (self.IS_DOCKER or self.APP_ENV in ("docker", "production")) else default_local_path
        )

        self.TILES_DEFAULT_CID = os.getenv("TILES_DEFAULT_CID", "abc123")
        self.S2_COLLECTION = os.getenv("S2_COLLECTION", "COPERNICUS/S2_HARMONIZED")
        self.TILE_CACHE_SECONDS = int(os.getenv("TILE_CACHE_SECONDS", "86400"))
        self.TILE_STALE_SECONDS = int(os.getenv("TILE_STALE_SECONDS", "604800"))

        self.CLIENT_IDS = os.getenv("CLIENT_IDS", "")

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        host = self.POSTGRES_SERVER
        port = f":{self.POSTGRES_PORT}" if self.POSTGRES_PORT else ""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{host}{port}/{self.POSTGRES_DB}"

    def get_gee_credentials_kwargs(self) -> dict:
        if self.GEE_KEY_JSON:
            return {"from_info": json.loads(self.GEE_KEY_JSON)}
        if self.GEE_KEY_B64:
            decoded = base64.b64decode(self.GEE_KEY_B64).decode("utf-8")
            return {"from_info": json.loads(decoded)}
        path = self.GOOGLE_APPLICATION_CREDENTIALS or self.GEE_KEY_PATH
        return {"from_file": path}

settings = Settings()
