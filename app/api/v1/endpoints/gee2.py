# app/api/gee_indices.py

import os
import datetime
from typing import List, Optional, Tuple

import ee
import requests
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from pydantic import BaseModel

router = APIRouter(
    prefix="/gee/indices",
    tags=["GEE Indices"]
)

# ========= Config & EE Init =========
PROJECT_ID = os.getenv("EARTHENGINE_PROJECT", None)
KEY_PATH = os.getenv("GEE_KEY_PATH", "D:\\Projects\\Projects\\58-AGRAS\\key.json")
TILES_DEFAULT_CID = os.getenv("TILES_DEFAULT_CID", "abc123")

if not KEY_PATH or not os.path.exists(KEY_PATH):
    raise RuntimeError("GEE_KEY_PATH no configurado o archivo no existe.")

credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/earthengine"],
    quota_project_id=PROJECT_ID,
)
ee.Initialize(credentials=credentials, project=PROJECT_ID)

# ========= Modelos =========
class IndexResponse(BaseModel):
    year: int
    month: int
    ndvi: Optional[float]
    ndmi: Optional[float]
    si: Optional[float]

class TimeSeriesResponse(BaseModel):
    dates: List[datetime.date]
    ndvi: List[Optional[float]]
    ndmi: List[Optional[float]]
    si: List[Optional[float]]

# ========= Helpers GEE =========
def mask_clouds(img: ee.Image) -> ee.Image:
    qa = img.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(mask)

def add_indices(img: ee.Image) -> ee.Image:
    green = img.select("B3")
    red   = img.select("B4")
    nir   = img.select("B8")
    swir1 = img.select("B11")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    si   = swir1.divide(green).rename("SI")

    return img.addBands([ndvi, ndmi, si])

def build_monthly_composites(aoi: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
    start = ee.Date(start_date)
    end   = ee.Date(end_date)
    start_year = ee.Number.parse(start.format('Y'))
    end_year   = ee.Number.parse(end.format('Y'))
    years  = ee.List.sequence(start_year, end_year)
    months = ee.List.sequence(1, 12)

    def per_image(y, m):
        y = ee.Number(y)
        m = ee.Number(m)
        m_start = ee.Date.fromYMD(y, m, 1)
        m_end   = m_start.advance(1, 'month')

        coll = (
            ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterDate(m_start, m_end)
            .filterBounds(aoi)
            .map(mask_clouds)
            .map(lambda i: i.select(["B3","B4","B8","B11"]).multiply(0.0001))
            .map(add_indices)
        )

        img = ee.Image(ee.Algorithms.If(
            coll.size().gt(0),
            coll.median().set({
                "year": y,
                "month": m,
                "system:time_start": m_start.millis()
            }),
            ee.Image(0).updateMask(ee.Image(0)).set({
                "empty": 1,
                "year": y,
                "month": m,
                "system:time_start": m_start.millis()
            })
        ))
        return img.clip(aoi)

    images_nested = years.map(lambda yy: months.map(lambda mm: per_image(yy, mm)))
    images = ee.List(images_nested).flatten()

    ic = ee.ImageCollection.fromImages(images)
    ic = ic.filter(ee.Filter.neq('empty', 1))
    ic = ic.filterDate(start, end)
    return ic

# --------- util: parsea mapid devolviendo (project_owner, map_id) ----------
def parse_mapid(raw_mapid: str) -> Tuple[Optional[str], str]:
    """
    raw_mapid puede venir como:
      - 'projects/<project_owner>/maps/<map_id>'
      - '<map_id>'
    Devuelve (project_owner, map_id). project_owner puede ser None si no venía en el string.
    """
    parts = raw_mapid.split("/")
    if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "maps":
        return parts[1], parts[3]
    return None, raw_mapid

# ========= Endpoints =========
@router.get("/monthly", response_model=List[IndexResponse], summary="Medias mensuales NDVI/NDMI/SI")
def get_monthly_indices(
    lon_min: float = Query(..., description="Longitud mínima AOI"),
    lat_min: float = Query(..., description="Latitud mínima AOI"),
    lon_max: float = Query(..., description="Longitud máxima AOI"),
    lat_max: float = Query(..., description="Latitud máxima AOI"),
    start: str = Query("2023-01-01", description="YYYY-MM-DD"),
    end: str = Query(datetime.date.today().isoformat(), description="YYYY-MM-DD"),
):
    aoi = ee.Geometry.Polygon(
        [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ],
        None,
        False,
    )

    coll = build_monthly_composites(aoi, start, end)
    size = coll.size().getInfo()
    if size == 0:
        return []

    imgs = coll.toList(size)
    results: List[IndexResponse] = []
    for i in range(size):
        img   = ee.Image(imgs.get(i))
        year  = int(ee.Number(img.get("year")).getInfo())
        month = int(ee.Number(img.get("month")).getInfo())
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10,
            maxPixels=1e9,
            tileScale=2,
        ).getInfo() or {}
        results.append(IndexResponse(
            year=year, month=month,
            ndvi=stats.get("NDVI"),
            ndmi=stats.get("NDMI"),
            si=stats.get("SI"),
        ))
    return results

@router.get("/timeseries", response_model=TimeSeriesResponse, summary="Serie temporal de índices")
def get_time_series(
    lon_min: float = Query(...),
    lat_min: float = Query(...),
    lon_max: float = Query(...),
    lat_max: float = Query(...),
    start: str = Query("2023-01-01"),
    end: str = Query(datetime.date.today().isoformat()),
):
    monthly = get_monthly_indices(lon_min, lat_min, lon_max, lat_max, start, end)
    dates = [datetime.date(m.year, m.month, 1) for m in monthly]
    return TimeSeriesResponse(
        dates=dates,
        ndvi=[m.ndvi for m in monthly],
        ndmi=[m.ndmi for m in monthly],
        si=[m.si for m in monthly],
    )

@router.get("/tile", summary="URL template (z/x/y) de tiles para un índice y mes")
def get_index_tile(
    request: Request,
    lon_min: float = Query(...),
    lat_min: float = Query(...),
    lon_max: float = Query(...),
    lat_max: float = Query(...),
    year: int = Query(..., description="Año, e.g., 2025"),
    month: int = Query(..., description="Mes 1–12"),
    index: str = Query("NDVI", description="NDVI | NDMI | SI"),
    palette: Optional[str] = Query(None, description="Hex o nombres separados por coma"),
    vmin: Optional[float] = Query(None),
    vmax: Optional[float] = Query(None),
):
    index = index.upper()
    if index not in ("NDVI", "NDMI", "SI"):
        raise HTTPException(status_code=400, detail="index debe ser NDVI, NDMI o SI")

    aoi = ee.Geometry.Polygon(
        [
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ],
        None,
        False,
    )

    d0 = datetime.date(year, month, 1)
    d1 = (d0 + datetime.timedelta(days=32)).replace(day=1)

    coll = (
        ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
        .filterDate(d0.isoformat(), d1.isoformat())
        .filterBounds(aoi)
        .map(mask_clouds)
        .map(lambda i: i.select(["B3","B4","B8","B11"]).multiply(0.0001))
        .map(add_indices)
    )
    if coll.size().getInfo() == 0:
        raise HTTPException(status_code=404, detail="No hay imágenes para el mes seleccionado")

    defaults = {
        "NDVI": {"min": -0.1, "max": 0.4, "palette": ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']},
        "NDMI": {"min": -0.2, "max": 0.2, "palette": ['#d73027', '#f7f7f7', '#4575b4']},
        "SI":   {"min":  1.0, "max": 2.5, "palette": ['#ffffff', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04']},
    }
    vis = defaults[index].copy()
    if palette:
        vis["palette"] = [p.strip() for p in palette.split(",") if p.strip()]
    if vmin is not None: vis["min"] = float(vmin)
    if vmax is not None: vis["max"] = float(vmax)

    median  = coll.median().select(index).clip(aoi)
    vis_img = median.visualize(min=vis["min"], max=vis["max"], palette=vis["palette"]).clip(aoi)

    # MapId (Cloud): puede venir con el proyecto dueño incluido
    mapid = vis_img.getMapId({"format": "png"})
    raw_mapid = str(mapid["mapid"])  # ej: 'projects/mi-proyecto/maps/<id>' OR '<id>'
    project_owner, just_id = parse_mapid(raw_mapid)
    map_project = project_owner or PROJECT_ID  # usa el dueño real si viene, si no tu PROJECT_ID

    # Construye URL del proxy incluyendo el proyecto dueño
    # ruta: /gee/indices/tiles/{project}/{mapid}/{z}/{x}/{y}.png
    path = request.app.url_path_for("proxy_tile", project=map_project, mapid=just_id, z=0, x=0, y=0)
    base = str(request.base_url).rstrip("/")
    template = f"{base}{path}".replace("/0/0/0.png", "/{z}/{x}/{y}.png")

    # Añade siempre cid (UrlTile no envía headers)
    client_id = request.headers.get("x-client-id") or request.headers.get("X-Client-ID")
    cid = client_id or TILES_DEFAULT_CID
    sep = "&" if "?" in template else "?"
    template = f"{template}{sep}cid={cid}"

    return JSONResponse({
        "tile_url_template": template,
        "vis": vis,
        "year": year,
        "month": month,
        "index": index,
        "mapid": {"mapid": just_id, "project": map_project},
    })

@router.get("/tiles/{project}/{mapid}/{z}/{x}/{y}.png", name="proxy_tile")
def proxy_tile(project: str, mapid: str, z: int, x: int, y: int, request: Request):
    """
    Proxy de tiles hacia EE Cloud API.
    Se usa el 'project' dueño del map (el que vino en raw_mapid) para evitar 403/404.
    """
    try:
        cid = request.query_params.get("cid")

        if not project:
            raise HTTPException(status_code=500, detail="Proyecto del mapa no determinado")

        # Refresca token OAuth (Service Account)
        scoped = credentials.with_scopes(["https://www.googleapis.com/auth/earthengine"])
        scoped.refresh(GoogleAuthRequest())

        headers = {
            "Authorization": f"Bearer {scoped.token}",
            "Accept": "image/png",
            "X-Goog-User-Project": project,   # cobra en el proyecto dueño del map
        }

        ee_tile_url = f"https://earthengine.googleapis.com/v1/projects/{project}/maps/{mapid}/tiles/{z}/{x}/{y}"
        r = requests.get(ee_tile_url, headers=headers, timeout=30)

        if r.status_code != 200:
            # Log claro para que veas el motivo exacto que devuelve Google
            print(f"[proxy_tile] EE error {r.status_code} for {ee_tile_url}: {r.text}")
            raise HTTPException(status_code=r.status_code, detail=r.text)

        return Response(
            content=r.content,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=604800"},
        )
    except HTTPException:
        raise
    except Exception as e:
        print("[proxy_tile] error:", str(e))
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
