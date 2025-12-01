# app/services/gee_services.py
from __future__ import annotations
import datetime
import threading
from typing import Dict, List, Optional, Tuple

import ee
import requests
import google.auth
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account

from app.core.config import settings

# ---- Estado de inicialización (thread-safe) ----
_EE_LOCK = threading.Lock()
_EE_INIT = False
_CREDS: Optional[Credentials] = None

RF_PROPS = [
    "NDVI","NDMI","SI","NDB12B7","TBI",
    "EVI","MSAVI2","NDRE","MSI",
    "VVdB","VH_VV",
]

EE_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/devstorage.read_write",  # si no usas GCS, puedes quitar esta
]

def _resolve_credentials() -> Tuple[Credentials, Optional[str]]:
    """
    Devuelve (credentials, project_id) en este orden de prioridad:
    - JSON inline (GEE_KEY_JSON / GEE_KEY_B64)
    - Archivo (GEE_KEY_PATH / GOOGLE_APPLICATION_CREDENTIALS) si existe
    - ADC (Cloud Run / gcloud ADC local)
    """
    kw = settings.get_gee_credentials_kwargs()

    if "from_info" in kw:
        creds = service_account.Credentials.from_service_account_info(
            kw["from_info"], scopes=EE_SCOPES
        )
        return creds, settings.EARTHENGINE_PROJECT

    if "from_file" in kw:
        creds = service_account.Credentials.from_service_account_file(
            kw["from_file"], scopes=EE_SCOPES
        )
        return creds, settings.EARTHENGINE_PROJECT

    # Fallback: ADC
    creds, project_id = google.auth.default(scopes=EE_SCOPES)
    project_id = settings.EARTHENGINE_PROJECT or project_id
    return creds, project_id

def ensure_ee_initialized() -> None:
    global _EE_INIT, _CREDS
    if _EE_INIT:
        return
    with _EE_LOCK:
        if _EE_INIT:
            return
        creds, project_id = _resolve_credentials()
        # 👇 Log útil: de dónde salen las credenciales
        origin = "ADC"
        try:
            from google.oauth2.service_account import Credentials as SACreds
            if isinstance(creds, SACreds):
                origin = "ServiceAccount"
        except Exception:
            pass
        print(f"[EE] Initializing with {origin}, project={project_id}")
        ee.Initialize(credentials=creds, project=project_id)
        _CREDS = creds
        _EE_INIT = True

def _get_access_token() -> str:
    """Devuelve un access token válido para llamar a la API de EE."""
    ensure_ee_initialized()
    assert _CREDS is not None
    # Refresca si hace falta
    _CREDS.refresh(GoogleAuthRequest())
    # Tras refresh, .token debe estar presente
    return getattr(_CREDS, "token", "")

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
        self.default_cid = settings.TILES_DEFAULT_CID

    # --- Consultas ---
    def monthly_indices(self, aoi: ee.Geometry, start: str, end: str) -> List[Dict]:
        """
        Devuelve [{year, month, NDVI, NDMI, SI}, ...] ordenado por fecha.
        Calcula estadísticas con un único getInfo() usando FeatureCollection.map.
        """
        ensure_ee_initialized()  # 👈 nos aseguramos aquí
        ic = build_monthly_composites(aoi, start, end)

        def to_feature(img: ee.Image):
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=aoi,
                scale=10,
                maxPixels=1e9,
                tileScale=2,
            )
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
        out: List[Dict] = []
        for f in feats:
            p = f.get("properties", {})
            out.append({
                "year": int(p.get("year")),
                "month": int(p.get("month")),
                "ndvi": p.get("NDVI"),
                "ndmi": p.get("NDMI"),
                "si":   p.get("SI"),
            })
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
        ensure_ee_initialized()  # 👈 nos aseguramos aquí

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

        token = _get_access_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "image/png",
        }
        # Factura a tu proyecto si corresponde
        if settings.EARTHENGINE_PROJECT:
            headers["X-Goog-User-Project"] = settings.EARTHENGINE_PROJECT

        url = f"https://earthengine.googleapis.com/v1/projects/{owner_project}/maps/{mapid}/tiles/{z}/{x}/{y}"
        return requests.get(url, headers=headers, timeout=30)


    def _build_comp_all(self, aoi: ee.Geometry, center_date: str) -> ee.Image:
        """
        Replica tu compAll: S2 (ventana ±3 días) + índices + S1 (VVdB, VH_VV)
        """
        ensure_ee_initialized()

        fecha_centro = ee.Date(center_date)
        fecha_ini = fecha_centro.advance(-3, "day")
        fecha_fin = fecha_centro.advance(4, "day")

        # ---- Sentinel-2 ----
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(fecha_ini, fecha_fin)
            .filterBounds(aoi)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 40))
        )

        def mask_s2(img):
            scl = img.select("SCL")
            bad = (
                scl.eq(3)
                .Or(scl.eq(7))
                .Or(scl.eq(8))
                .Or(scl.eq(9))
                .Or(scl.eq(10))
            )
            return img.updateMask(bad.Not())

        comp = s2.map(mask_s2).median().clip(aoi)

        # ----- índices ópticos -----
        b2  = comp.select("B2")
        b3  = comp.select("B3")
        b4  = comp.select("B4")
        b5  = comp.select("B5")
        b7  = comp.select("B7")
        b8  = comp.select("B8")
        b8a = comp.select("B8A")
        b11 = comp.select("B11")
        b12 = comp.select("B12")

        ndvi = comp.normalizedDifference(["B8","B4"]).rename("NDVI")
        ndmi = comp.normalizedDifference(["B8","B11"]).rename("NDMI")
        si   = b11.divide(b3).rename("SI")
        ndb12b7 = comp.normalizedDifference(["B12","B7"]).rename("NDB12B7")

        denom_tbi = b3.subtract(b11)
        tbi = (
            b12.subtract(b3)
               .divide(denom_tbi)
               .updateMask(denom_tbi.neq(0))
               .rename("TBI")
        )

        evi = (
            b8.subtract(b4).multiply(2.5)
               .divide(
                   b8.add(b4.multiply(6))
                     .subtract(b2.multiply(7.5))
                     .add(1)
               )
               .rename("EVI")
        )

        msavi2 = (
            b8.multiply(2).add(1)
               .subtract(
                   b8.multiply(2).add(1).pow(2)
                     .subtract(b8.subtract(b4).multiply(8))
                     .sqrt()
               )
               .multiply(0.5)
               .rename("MSAVI2")
        )

        ndre = comp.normalizedDifference(["B8A","B5"]).rename("NDRE")
        msi  = b11.divide(b8).rename("MSI")

        comp_idx = comp.addBands([
            ndvi, ndmi, si, ndb12b7, tbi,
            evi, msavi2, ndre, msi,
        ])

        # ----- Sentinel-1 -----
        s1_ini_06 = fecha_centro.advance(-6, "day")
        s1_fin_06 = fecha_centro.advance( 6, "day")
        s1_ini_30 = fecha_centro.advance(-30, "day")
        s1_fin_30 = fecha_centro.advance( 30, "day")

        def s1_between(start, end):
            return (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterDate(start, end)
                .filterBounds(aoi)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            )

        s1col06 = s1_between(s1_ini_06, s1_fin_06)
        s1col30 = s1_between(s1_ini_30, s1_fin_30)
        s1_use = ee.ImageCollection(
            ee.Algorithms.If(s1col06.size().gt(0), s1col06, s1col30)
        )
        has_s1 = s1_use.size().gt(0)

        s1_img = ee.Image(
            ee.Algorithms.If(
                has_s1,
                s1_use.median().clip(aoi),
                ee.Image.constant([1e-6, 1e-6]).rename(["VV","VH"]).clip(aoi),
            )
        )

        vv = s1_img.select("VV")
        vh = s1_img.select("VH")
        vv_safe = vv.where(vv.lte(0), 1e-6)
        vv_db = vv_safe.log10().multiply(10).rename("VVdB")
        vh_vv = vh.divide(vv_safe).rename("VH_VV")

        comp_all = comp_idx.addBands([vv_db, vh_vv])

        return comp_all.clip(aoi)

    def _salinity_rf_image(self, aoi: ee.Geometry, center_date: str) -> ee.Image:
        ensure_ee_initialized()

        asset_id = settings.SALINITY_RF_TRAINING_ASSET
        if not asset_id:
            raise RuntimeError(
                "Config SALINITY_RF_TRAINING_ASSET no definida. "
                "Configura la variable de entorno con el ID del asset de entrenamiento en Earth Engine."
            )

        print("[salinityRF] usando asset:", asset_id)
        train_fc = ee.FeatureCollection(asset_id)
        print("[salinityRF] tamaño training:", train_fc.size().getInfo())

        comp_all = self._build_comp_all(aoi, center_date)

        rf = (
            ee.Classifier.smileRandomForest(numberOfTrees=50)
            .setOutputMode("REGRESSION")
        )
        rf_trained = rf.train(
            features=train_fc,
            classProperty="CE_mS",
            inputProperties=RF_PROPS,
        )

        ce_rf_img = comp_all.select(RF_PROPS).classify(rf_trained).rename("CE_RF")
        return ce_rf_img
    
    def salinity_rf_tile_template(
            self,
            lon_min: float,
            lat_min: float,
            lon_max: float,
            lat_max: float,
            center_date: str,
            palette_csv: Optional[str],
            vmin: Optional[float],
            vmax: Optional[float],
            base_url: str,
            proxy_path_builder,
            client_id_header: Optional[str],
        ) -> Dict:
            aoi = make_aoi(lon_min, lat_min, lon_max, lat_max)
            ce_rf_img = self._salinity_rf_image(aoi, center_date)

            vis = {
                "min": 0.0,
                "max": 0.6,
                "palette": ["#003366","#66ccff","#ffffcc","#ffcc66","#cc6600"],
            }
            if vmin is not None:
                vis["min"] = float(vmin)
            if vmax is not None:
                vis["max"] = float(vmax)
            if palette_csv:
                vis["palette"] = [p.strip() for p in palette_csv.split(",") if p.strip()]

            vis_img = ce_rf_img.visualize(
                min=vis["min"], max=vis["max"], palette=vis["palette"]
            ).clip(aoi)

            mapid = vis_img.getMapId({"format": "png"})
            raw_mapid = str(mapid["mapid"])
            project_owner, just_id = parse_mapid(raw_mapid)
            owner = project_owner or "earthengine-legacy"

            path_0 = proxy_path_builder(project=owner, mapid=just_id, z=0, x=0, y=0)
            template = f"{base_url.rstrip('/')}{path_0}".replace(
                "/0/0/0.png", "/{z}/{x}/{y}.png"
            )

            cid = client_id_header or settings.TILES_DEFAULT_CID
            sep = "&" if "?" in template else "?"
            template = f"{template}{sep}cid={cid}"

            dt = datetime.date.fromisoformat(center_date)

            return {
                "tile_url_template": template,
                "vis": vis,
                "year": dt.year,
                "month": dt.month,
                "index": "CE_RF",
                "mapid": {"mapid": just_id, "project": owner},
            }