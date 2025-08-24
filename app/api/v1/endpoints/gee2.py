# app/api/gee_indices.py

import os
import datetime
from typing import List, Optional
from fastapi.responses import JSONResponse

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
import ee
from google.oauth2 import service_account

from app.db.session import get_db
from app.utils.response import create_response

router = APIRouter(
    prefix="/gee/indices",
    tags=["GEE Indices"]
)

# ————— Inicialización de EE —————

PROJECT_ID = os.getenv("EARTHENGINE_PROJECT", None)
KEY_PATH = os.getenv("GEE_KEY_PATH", "D:\\Projects\\Projects\\58-AGRAS\\key.json")

# Carga credenciales
credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/earthengine"],
    quota_project_id=PROJECT_ID,
)
ee.Initialize(credentials=credentials, project=PROJECT_ID)

# ————— Models —————

class IndexResponse(BaseModel):
    year: int
    month: int
    ndvi: float
    ndmi: float
    si: float

class TimeSeriesResponse(BaseModel):
    dates: List[datetime.date]
    ndvi: List[float]
    ndmi: List[float]
    si: List[float]

# ————— Funciones auxiliares de GEE —————

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

def build_monthly_composites(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str
) -> ee.ImageCollection:
    """Devuelve una ImageCollection con medianas mensuales de NDVI, NDMI y SI."""
    start = datetime.datetime.fromisoformat(start_date)
    end   = datetime.datetime.fromisoformat(end_date)
    years = list(range(start.year, end.year + 1))
    months = list(range(1, 13))

    imgs = []
    for y in years:
        for m in months:
            # Definimos rango mensual
            d0 = datetime.date(y, m, 1)
            d1 = (d0 + datetime.timedelta(days=32)).replace(day=1)
            coll = (
                ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterDate(d0.isoformat(), d1.isoformat())
                .filterBounds(aoi)
                .map(mask_clouds)
                .map(lambda i: i.select(["B3","B4","B8","B11"]).multiply(0.0001))
                .map(add_indices)
            )

            # Usamos ee.Algorithms.If en vez de .conditional()
            img_or_null = ee.Algorithms.If(
                coll.size().gt(0),
                coll.median()
                    .set("year", y)
                    .set("month", m)
                    .set("system:time_start", ee.Date.fromYMD(y, m, 1).millis()),
                None
            )
            # ee.Algorithms.If devuelve un ee.Image cuando hay datos, o None; 
            # envolvemos en ee.Image() solo si no es None
            imgs.append(ee.Image(img_or_null) if isinstance(img_or_null, ee.Image) else img_or_null)

    # Filtramos los None y construimos la ImageCollection
    return ee.ImageCollection([img for img in imgs if img is not None])


@router.get(
    "/monthly",
    response_model=List[IndexResponse],
    summary="Valores medios mensuales de NDVI, NDMI y SI"
)
async def get_monthly_indices(
    lon_min: float = Query(-71.88100053463857, description="Longitud mínima AOI"),
    lat_min: float = Query(-16.73354802470063, description="Latitud mínima AOI"),
    lon_max: float = Query(-71.87370492611807, description="Longitud máxima AOI"),
    lat_max: float = Query(-16.73001355926052, description="Latitud máxima AOI"),
    start: str = Query("2023-01-01", description="Fecha inicio (YYYY-MM-DD)"),
    end: str = Query(datetime.date.today().isoformat(), description="Fecha fin (YYYY-MM-DD)")
):
    # Usar la AOI predefinida si no cambia
    aoi = ee.Geometry.Polygon([
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ])
    coll = build_monthly_composites(aoi, start, end)
    features = coll.toList(coll.size())

    results = []
    for i in range(coll.size().getInfo()):
        img = ee.Image(features.get(i))
        year  = img.get("year").getInfo()
        month = img.get("month").getInfo()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=10
        ).getInfo()
        results.append(IndexResponse(
            year=year,
            month=month,
            ndvi=stats.get("NDVI", None),
            ndmi=stats.get("NDMI", None),
            si=stats.get("SI", None),
        ))

    return results

@router.get(
    "/timeseries",
    response_model=TimeSeriesResponse,
    summary="Serie temporal completa de los índices"
)
async def get_time_series(
    lon_min: float = Query(-71.88100053463857),
    lat_min: float = Query(-16.73354802470063),
    lon_max: float = Query(-71.87370492611807),
    lat_max: float = Query(-16.73001355926052),
    start: str = Query("2023-01-01"),
    end: str = Query(datetime.date.today().isoformat())
):
    monthly = await get_monthly_indices(lon_min, lat_min, lon_max, lat_max, start, end)
    dates = [datetime.date(m.year, m.month, 1) for m in monthly]
    return TimeSeriesResponse(
        dates=dates,
        ndvi=[m.ndvi for m in monthly],
        ndmi=[m.ndmi for m in monthly],
        si=[m.si for m in monthly],
    )

@router.get(
    "/tile",
    summary="Devuelve URL de tiles para un índice (NDVI/NDMI/SI) en un mes dado",
)
async def get_index_tile(
    lon_min: float = Query(...),
    lat_min: float = Query(...),
    lon_max: float = Query(...),
    lat_max: float = Query(...),
    year: int = Query(..., description="Año, e.g. 2025"),
    month: int = Query(..., description="Mes 1-12"),
    index: str = Query("NDVI", description="NDVI | NDMI | SI"),
    palette: Optional[str] = Query(None, description="Paleta opcional coma-separada"),
    # opcionales para control visual
    vmin: Optional[float] = Query(None),
    vmax: Optional[float] = Query(None),
):
    """
    Devuelve un template de URL para tiles (z/x/y) renderizados por Earth Engine.
    """
    index = index.upper()
    if index not in ("NDVI", "NDMI", "SI"):
        raise HTTPException(status_code=400, detail="index debe ser NDVI, NDMI o SI")

    # AOI
    aoi = ee.Geometry.Polygon([
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ])

    # Rango de fechas del mes
    d0 = datetime.date(year, month, 1)
    d1 = (d0 + datetime.timedelta(days=32)).replace(day=1)
    start_iso = d0.isoformat()
    end_iso = d1.isoformat()

    # construir colección y calcular mediana mensual con índices
    coll = (
    ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    .filterDate(start_iso, end_iso)
    .filterBounds(aoi)
    .map(mask_clouds)
    .map(lambda i: i.select(["B3","B4","B8","B11"]).multiply(0.0001))
    .map(add_indices)
)

    size = coll.size().getInfo()
    if size == 0:
        raise HTTPException(status_code=404, detail="No hay imágenes para el mes seleccionado")

    # median y CLIP al AOI (critical: clip para que el tile solo pinte el área seleccionada)
    img = coll.median().select(index).clip(aoi)

    # visualización por defecto (puedes sobreescribir con palette/vmin/vmax en query params)
    defaults = {
        "NDVI": {"min": -0.1, "max": 0.4, "palette": ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']},
        "NDMI": {"min": -0.2, "max": 0.2, "palette": ['#d73027', '#f7f7f7', '#4575b4']},
        "SI":   {"min": 1.0,  "max": 2.5, "palette": ['#ffffff', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04']},
    }
    vis = defaults[index].copy()
    if palette:
        vis["palette"] = palette.split(",")
    if vmin is not None:
        vis["min"] = float(vmin)
    if vmax is not None:
        vis["max"] = float(vmax)

    # Opcional: forzar formato png para transparencia
    vis["format"] = "png"

    # obtener mapid + token
    mapid = img.getMapId(vis)

    # template con token (lista para {z}/{x}/{y})
    tile_url = f"https://earthengine.googleapis.com/map/{mapid['mapid']}/{{z}}/{{x}}/{{y}}?token={mapid['token']}"

    return JSONResponse({
        "tile_url_template": tile_url,
        "vis": vis,
        "year": year,
        "month": month,
        "index": index,
        "mapid": {"mapid": mapid["mapid"]}
    })