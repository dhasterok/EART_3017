from pathlib import Path

DATASETS = {
    "srtm15": {
        "path": Path("../../data/topography/SRTM15_V2.7.nc"),
        "var": "z",
        "units": "m",
        "centering": "cell",
    },
    "sstopo2": {
        "path": Path("../../data/grids/sstopo2.nc"),
        "var": "z",
        "units": "m",
        "centering": "cell",
    },
    "sfage": {
        "path": Path("../../data/grids/sfage.nc"),
        "var": "age",
        "units": "Ma",
        "centering": "cell",
    },
    "plates": {
        "path": Path("../../data/grids/plates.nc"),
        "var": "plate_id",
        "units": None,
        "centering": "cell",
    },
    # Extend freely
}