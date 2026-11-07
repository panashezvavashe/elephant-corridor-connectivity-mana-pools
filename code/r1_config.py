"""Shared configuration. Every script imports this module first.

Importing it caps the BLAS/OpenMP thread pools and lowers the process priority (Windows),
so the long model fits and Circuitscape runs do not saturate a laptop CPU. All paths are
relative to the repository root.
"""
import os
from pathlib import Path


def _set_proj():
    """Use the PROJ database bundled with rasterio (avoids clashes with other PROJ installs)."""
    try:
        import importlib.util
        p = os.path.join(os.path.dirname(importlib.util.find_spec('rasterio').origin), 'proj_data')
        if os.path.isdir(p):
            os.environ['PROJ_LIB'] = os.environ['PROJ_DATA'] = p
    except Exception:
        pass


_set_proj()

N_JOBS = int(os.environ.get('N_JOBS', 3))
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ.setdefault(_v, str(N_JOBS))


def lower_priority():
    try:
        import ctypes
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, 0x00004000)  # BELOW_NORMAL
    except Exception:
        pass


lower_priority()

SEED = 42
CRS = 'EPSG:32736'  # WGS 84 / UTM zone 36S

REPO = Path(__file__).resolve().parents[1].as_posix()
DATA_DIR = REPO + '/data'
BOUNDARY_DIR = DATA_DIR + '/boundaries'
AKDE_DIR = DATA_DIR + '/akde95'            # AKDE 95% home ranges per collar (Objective 1)
OUT_DATA = DATA_DIR + '/derived'           # fitted models, settings, and (git-ignored) GPS-derived files
OUT = REPO + '/results'
OUT_TAB = OUT + '/tables'
OUT_FIG = OUT + '/figures'
OUT_RAS = OUT + '/rasters'
for _d in (OUT_DATA, OUT, OUT_TAB, OUT_FIG, OUT_RAS):
    os.makedirs(_d, exist_ok=True)

PREDICTOR_TIF = DATA_DIR + '/predictors/predictors_30m_stack.tif'  # built by build_predictor_stack.py
GPS_SHP = DATA_DIR + '/gps/GPS_points.shp'                         # not distributed, see data/gps/README.md
STUDY_SHP = BOUNDARY_DIR + '/study_area.gpkg'
CORE_SHP = DATA_DIR + '/core_areas/core_areas_dBBMM50.gpkg'        # dBBMM 50% cores, Objective 1

# Collar 4557 is excluded from every analysis.
IDS = [3905, 3906, 3907, 4141, 4142, 4501, 4556, 4558, 4559, 4560]
EXCLUDED_IDS = [4557]
SEX = {3905: 'M', 3906: 'M', 3907: 'M', 4141: 'F', 4142: 'M',
       4501: 'M', 4556: 'M', 4558: 'F', 4559: 'F', 4560: 'M'}

N_PRESENCE = 5000
N_ABSENCE = 5000
ABSENCE_EXCLUSION_M = 500
N_VALIDATION_FIXES = 1000

BAND_NAMES = ['Elevation', 'Slope', 'TPI', 'TRI', 'Dist_Perm_Water',
              'Dist_Season_Water', 'Wet_NDVI', 'Dry_NDVI',
              'LandCover', 'Dist_Built_Up', 'Nighttime_Lights']
CONTINUOUS = [b for b in BAND_NAMES if b != 'LandCover']
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 80, 90]  # ESA WorldCover classes present
LULC_COLS = [f'LULC_{c}' for c in LULC_CLASSES]
FEATURES = CONTINUOUS + LULC_COLS
