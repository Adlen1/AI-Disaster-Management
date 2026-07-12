# scripts/utils/grid.py
import geopandas as gpd
import numpy as np
from rasterio.transform import from_bounds
from dataclasses import dataclass


@dataclass
class AlgeriaGrid:
    transform: object
    width:     int
    height:    int
    bounds:    tuple
    res_m:     float
    crs:       str = "EPSG:4326"

    def shape(self):
        return (self.height, self.width)

    def empty_array(self, fill=np.nan):
        return np.full(self.shape(), fill, dtype=np.float32)


def get_algeria_grid(boundary_path, target_res_m=1000) -> AlgeriaGrid:
    algeria        = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = algeria.total_bounds
    target_res_deg = target_res_m / 111000
    width          = round((maxx - minx) / target_res_deg)
    height         = round((maxy - miny) / target_res_deg)
    transform      = from_bounds(minx, miny, maxx, maxy, width, height)

    return AlgeriaGrid(
        transform=transform,
        width=width,
        height=height,
        bounds=(minx, miny, maxx, maxy),
        res_m=target_res_m
    )