# app/schemas/gee.py
from datetime import date
from typing import List, Optional
from pydantic import BaseModel

class IndexResponse(BaseModel):
    year: int
    month: int
    ndvi: Optional[float]
    ndmi: Optional[float]
    si: Optional[float]

class TimeSeriesResponse(BaseModel):
    dates: List[date]
    ndvi: List[Optional[float]]
    ndmi: List[Optional[float]]
    si: List[Optional[float]]

class TileTemplateResponse(BaseModel):
    tile_url_template: str
    vis: dict
    year: int
    month: int
    index: str
    mapid: dict
