# app/services/gee_service.py
from __future__ import annotations
import datetime
from typing import Dict, List, Optional, Tuple

import ee
import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest

from app.core.config import settings

# ---------- Inicialización de EE ----------
def _init_ee():
    kw = settings.get_gee_credentials_kwargs()
    if "from_info" in kw:
        creds = service_account.Credentials.from_service_account_info(
            kw["from_info"],
            scopes=["https://www.googleapis.com/auth/earthengine"],
            quota_project_id=settings.EARTHENGINE_PROJECT,
        )
    else:
        creds = service_account.Credentials.from_service_account_file(
            kw["from_file"],
            scopes=["https://www.googleapis.com/auth/earthengine"],
            quota_project_id=settings.EARTHENGINE_PROJECT,
        )
    ee.Initialize(credentials=creds, project=settings.EARTHENGINE_PROJECT)
    return creds

# Guardamos las credenciales de SA para refrescar en el proxy
_CREDENTIALS: service_account.Credentials = _init_ee()

# ---------- Helpers GEE puros ----------
def make_aoi(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> ee.Geometry:
    return ee.Geometry.Polygon(
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
            ee.ImageCollection(settings.S2_COLLECTION)
            .filterDate(m_start, m_end)
            .filterBounds(aoi)
            .map(mask_clouds)
            .map(lambda i: i.select(["B3","B4","B8","B11"]).multiply(0.0001))
            .map(add_indices)
        )

        img = ee.Image(ee.Algorithms.If(
            coll.size().gt(0),
            coll.median().set({"year": y, "month": m, "system:time_start": m_start.millis()}),
            ee.Image(0).updateMask(ee.Image(0)).set({
                "empty": 1, "year": y, "month": m, "system:time_start": m_start.millis()
            })
        ))
        return img.clip(aoi)

    images_nested = years.map(lambda yy: months.map(lambda mm: per_image(yy, mm)))
    images = ee.List(images_nested).flatten()

    ic = ee.ImageCollection.fromImages(images)
    ic = ic.filter(ee.Filter.neq('empty', 1))
    ic = ic.filterDate(start, end)
    return ic

def parse_mapid(raw_mapid: str) -> Tuple[Optional[str], str]:
    parts = raw_mapid.split("/")
    if len(parts) >= 4 and parts[0] == "projects" and parts[2] == "maps":
        return parts[1], parts[3]
    return None, raw_mapid

# ---------- Servicio de alto nivel ----------
class GEEService:
    def __init__(self) -> None:
        # EE ya fue inicializado a nivel módulo; guardamos settings
        self.default_cid = settings.TILES_DEFAULT_CID

    # --- Consultas ---
    def monthly_indices(self, aoi: ee.Geometry, start: str, end: str) -> List[Dict]:
        """
        Devuelve [{year, month, NDVI, NDMI, SI}, ...] ordenado por fecha.
        Calcula estadísticas con un único getInfo() usando FeatureCollection.map.
        """
        ic = build_monthly_composites(aoi, start, end)

        def to_feature(img: ee.Image):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e9,
                tileScale=2,
            )
            # Guardamos propiedades de fecha + medias como atributos del Feature
            return ee.Feature(None, {
                "year": img.get("year"),
                "month": img.get("month"),
                "NDVI": stats.get("NDVI"),
                "NDMI": stats.get("NDMI"),
                "SI":   stats.get("SI"),
            })

        fc = ee.FeatureCollection(ic.map(to_feature))
        data = fc.getInfo()  # único getInfo
        feats = data.get("features", [])
        out = []
        for f in feats:
            p = f.get("properties", {})
            out.append({
                "year": int(p.get("year")),
                "month": int(p.get("month")),
                "ndvi": p.get("NDVI"),
                "ndmi": p.get("NDMI"),
                "si":   p.get("SI"),
            })
        # Aseguramos orden por año/mes
        out.sort(key=lambda r: (r["year"], r["month"]))
        return out

    def tile_template(
        self,
        lon_min: float, lat_min: float, lon_max: float, lat_max: float,
        year: int, month: int, index: str,
        palette_csv: Optional[str], vmin: Optional[float], vmax: Optional[float],
        base_url: str, proxy_path_builder,
        client_id_header: Optional[str],
    ) -> Dict:
        index = index.upper()
        if index not in ("NDVI", "NDMI", "SI"):
            raise ValueError("index debe ser NDVI, NDMI o SI")

        if month < 1 or month > 12:
            raise ValueError("month debe estar entre 1 y 12")

        if vmin is not None and vmax is not None and vmin >= vmax:
            raise ValueError("vmin debe ser menor que vmax")

        aoi = make_aoi(lon_min, lat_min, lon_max, lat_max)
        d0 = datetime.date(year, month, 1)
        d1 = (d0 + datetime.timedelta(days=32)).replace(day=1)

        coll = (
            ee.ImageCollection(settings.S2_COLLECTION)
            .filterDate(d0.isoformat(), d1.isoformat())
            .filterBounds(aoi)
            .map(mask_clouds)
            .map(lambda i: i.select(["B3","B4","B8","B11"]).multiply(0.0001))
            .map(add_indices)
        )
        if coll.size().getInfo() == 0:
            raise LookupError("No hay imágenes para el mes seleccionado")

        defaults = {
            "NDVI": {"min": -0.1, "max": 0.4, "palette": ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']},
            "NDMI": {"min": -0.2, "max": 0.2, "palette": ['#d73027', '#f7f7f7', '#4575b4']},
            "SI":   {"min":  1.0, "max": 2.5, "palette": ['#ffffff', '#fff7bc', '#fee391', '#fec44f', '#fe9929', '#ec7014', '#cc4c02', '#8c2d04']},
        }
        vis = defaults[index].copy()

        if palette_csv:
            vis["palette"] = [p.strip() for p in palette_csv.split(",") if p.strip()]
        if vmin is not None:
            vis["min"] = float(vmin)
        if vmax is not None:
            vis["max"] = float(vmax)

        median  = coll.median().select(index).clip(aoi)
        vis_img = median.visualize(min=vis["min"], max=vis["max"], palette=vis["palette"]).clip(aoi)

        mapid = vis_img.getMapId({"format": "png"})
        raw_mapid = str(mapid["mapid"])  # 'projects/<owner>/maps/<id>' o '<id>'
        project_owner, just_id = parse_mapid(raw_mapid)
        owner = project_owner or "earthengine-legacy"
        path_0 = proxy_path_builder(project=owner, mapid=just_id, z=0, x=0, y=0)

        # path proxy (con placeholders z/x/y reemplazables)
        template = f"{base_url.rstrip('/')}{path_0}".replace("/0/0/0.png", "/{z}/{x}/{y}.png")

        cid = client_id_header or settings.TILES_DEFAULT_CID
        sep = "&" if "?" in template else "?"
        template = f"{template}{sep}cid={cid}"

        return {
            "tile_url_template": template,
            "vis": vis,
            "year": year,
            "month": month,
            "index": index,
            "mapid": {"mapid": just_id, "project": owner},
        }

    # --- Proxy de tiles ---
    def fetch_tile(self, owner_project: str, mapid: str, z: int, x: int, y: int) -> requests.Response:
        if not owner_project:
            raise ValueError("Proyecto owner del mapa no determinado")

        scoped = _CREDENTIALS.with_scopes(["https://www.googleapis.com/auth/earthengine"])
        scoped.refresh(GoogleAuthRequest())

        billing_project = settings.EARTHENGINE_PROJECT  # 👈 tu proyecto (habilitado para EE)
        headers = {
            "Authorization": f"Bearer {scoped.token}",
            "Accept": "image/png",
        }
        # Solo añade el header si tienes proyecto de facturación configurado
        if billing_project:
            headers["X-Goog-User-Project"] = billing_project  # ✅ factura a TU proyecto

        url = f"https://earthengine.googleapis.com/v1/projects/{owner_project}/maps/{mapid}/tiles/{z}/{x}/{y}"
        return requests.get(url, headers=headers, timeout=30)
