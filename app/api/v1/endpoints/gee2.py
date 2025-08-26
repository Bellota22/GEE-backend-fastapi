# app/api/v1/endpoints/gee2.py
import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from app.schemas.gee import IndexResponse, TimeSeriesResponse, TileTemplateResponse
from app.services.gee_service import GEEService, make_aoi
from app.core.config import settings

router = APIRouter(prefix="/gee/indices", tags=["GEE Indices"])
service = GEEService()

# --------- Endpoints ---------

@router.get("/monthly", response_model=List[IndexResponse], summary="Medias mensuales NDVI/NDMI/SI")
def get_monthly_indices(
    lon_min: float = Query(..., description="Longitud mínima AOI"),
    lat_min: float = Query(..., description="Latitud mínima AOI"),
    lon_max: float = Query(..., description="Longitud máxima AOI"),
    lat_max: float = Query(..., description="Latitud máxima AOI"),
    start: str = Query("2023-01-01", description="YYYY-MM-DD"),
    end: str = Query(datetime.date.today().isoformat(), description="YYYY-MM-DD"),
):
    try:
        aoi = make_aoi(lon_min, lat_min, lon_max, lat_max)
        data = service.monthly_indices(aoi, start, end)
        return [IndexResponse(**d) for d in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando mensuales: {e}")

@router.get("/timeseries", response_model=TimeSeriesResponse, summary="Serie temporal de índices")
def get_time_series(
    lon_min: float = Query(...),
    lat_min: float = Query(...),
    lon_max: float = Query(...),
    lat_max: float = Query(...),
    start: str = Query("2023-01-01"),
    end: str = Query(datetime.date.today().isoformat()),
):
    try:
        aoi = make_aoi(lon_min, lat_min, lon_max, lat_max)
        monthly = service.monthly_indices(aoi, start, end)
        dates = [datetime.date(m["year"], m["month"], 1) for m in monthly]
        return TimeSeriesResponse(
            dates=dates,
            ndvi=[m["ndvi"] for m in monthly],
            ndmi=[m["ndmi"] for m in monthly],
            si=[m["si"] for m in monthly],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en timeseries: {e}")

@router.get("/tile", response_model=TileTemplateResponse, summary="URL template (z/x/y) de tiles para un índice y mes")
def get_index_tile(
    request: Request,
    lon_min: float = Query(...),
    lat_min: float = Query(...),
    lon_max: float = Query(...),
    lat_max: float = Query(...),
    year: int = Query(..., description="Año, e.g., 2025"),
    month: int = Query(..., ge=1, le=12, description="Mes 1–12"),
    index: str = Query("NDVI", description="NDVI | NDMI | SI"),
    palette: Optional[str] = Query(None, description="Hex o nombres separados por coma"),
    vmin: Optional[float] = Query(None),
    vmax: Optional[float] = Query(None),
):
    try:
        payload = service.tile_template(
            lon_min, lat_min, lon_max, lat_max,
            year, month, index,
            palette_csv=palette, vmin=vmin, vmax=vmax,
            base_url=str(request.base_url),
            proxy_path_builder=lambda **kw: request.app.url_path_for("proxy_tile", **kw),
            client_id_header=request.headers.get("x-client-id") or request.headers.get("X-Client-ID"),
        )
        return JSONResponse(payload)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except LookupError as le:
        raise HTTPException(status_code=404, detail=str(le))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparando tiles: {e}")

@router.get("/tiles/{project}/{mapid}/{z}/{x}/{y}.png", name="proxy_tile")
def proxy_tile(project: str, mapid: str, z: int, x: int, y: int, request: Request):
    """
    Proxy de tiles hacia EE Cloud API.
    Cobra contra el proyecto dueño del map (X-Goog-User-Project).
    """
    try:
        r = service.fetch_tile(project, mapid, z, x, y)
        if r.status_code != 200:
            # Log del motivo original de Google
            print(f"[proxy_tile] EE error {r.status_code}: {r.text}")
            raise HTTPException(status_code=r.status_code, detail=r.text)

        return Response(
            content=r.content,
            media_type="image/png",
            headers={
                "Cache-Control": f"public, max-age={settings.TILE_CACHE_SECONDS}, stale-while-revalidate={settings.TILE_STALE_SECONDS}"
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print("[proxy_tile] error:", str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
