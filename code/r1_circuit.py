"""Step 4: Circuitscape 4 runs on the revised resistance surfaces.

Usage: python r1_circuit.py <MODEL> [--factor 3] [--clip-to-study] [--window xmin ymin xmax ymax] [--label NAME]

Default reproduces the submitted configuration (block-average 30 m -> 90 m, cores
rasterized on the same grid, pairwise mode, 8 neighbours, average resistance, cg+amg).
  --clip-to-study   cells outside the 5,742 km2 study polygon set to NoData (#14:
                    compares against the default, which uses the full predictor rectangle)
  --window          restrict to a sub-window (#8: 30 m vs 90 m comparison)
Circuitscape runs at below-normal priority to limit heat.
"""
import argparse
import os
import subprocess
import time

import r1_config as C
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize, geometry_mask
from rasterio.windows import from_bounds

CS_EXE = os.environ.get('CIRCUITSCAPE_EXE', r'C:/Program Files/Circuitscape/cs_run.exe')  # Circuitscape 4
BELOW_NORMAL = 0x00004000


def write_asc(path, arr, transform, fmt, nodata=True):
    h, w = arr.shape
    with open(path, 'w') as f:
        f.write(f'ncols         {w}\nnrows         {h}\n')
        f.write(f'xllcorner     {transform.c:.6f}\nyllcorner     {transform.f + transform.e * h:.6f}\n')
        f.write(f'cellsize      {abs(transform.a):.6f}\n')
        if nodata:
            f.write('NODATA_value  -9999\n')
        np.savetxt(f, arr, fmt=fmt)


def build_inputs(model, factor, clip, window, out_dir, surface=''):
    src_tif = f'{C.OUT_RAS}/{model.lower()}_resistance{surface}.tif'
    with rasterio.open(src_tif) as src:
        win = from_bounds(*window, transform=src.transform).round_offsets().round_lengths() if window else \
            rasterio.windows.Window(0, 0, src.width, src.height)
        h, w = int(win.height) // factor, int(win.width) // factor
        data = src.read(1, window=win, out_shape=(h, w), resampling=Resampling.average,
                        masked=True).filled(np.nan).astype(float)
        t = src.window_transform(win) * src.transform.scale(win.width / w, win.height / h)
    if surface:  # alternative transforms: gaps (none expected) take the surface maximum
        data = np.where(np.isnan(data), np.nanmax(data), data)
    else:
        data = np.clip(np.where(np.isnan(data), 0.99, data), 0.001, None)  # as circuit.py
    if clip:
        study = gpd.read_file(C.STUDY_SHP).to_crs(C.CRS)
        inside = geometry_mask(study.geometry, out_shape=data.shape, transform=t, invert=True)
        data = np.where(inside, data, -9999)
    cores = gpd.read_file(C.CORE_SHP).to_crs(C.CRS)
    idcol = 'elephant_i' if 'elephant_i' in cores.columns else 'elephant_id'
    cores = cores.set_index(idcol).loc[C.IDS].reset_index()
    nodes = rasterize(((g, i + 1) for i, g in enumerate(cores.geometry)), out_shape=data.shape,
                      transform=t, fill=0, dtype='int32')
    if clip:
        nodes = np.where(data == -9999, 0, nodes)
    present = sorted(int(v) for v in np.unique(nodes) if v > 0)
    res_asc, node_asc = f'{out_dir}/resistance.asc', f'{out_dir}/focal_nodes.asc'
    write_asc(res_asc, data, t, '%.6f')
    write_asc(node_asc, nodes, t, '%d', nodata=False)
    meta = dict(driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype='float32',
                crs=C.CRS, transform=t, nodata=-9999, compress='deflate')
    with rasterio.open(f'{out_dir}/resistance.tif', 'w', **meta) as d:
        d.write(data.astype('float32'), 1)
    with rasterio.open(f'{out_dir}/focal_nodes.tif', 'w', **{**meta, 'dtype': 'int32', 'nodata': 0}) as d:
        d.write(nodes, 1)
    return res_asc, node_asc, t, data.shape, present


def write_ini(res_asc, node_asc, prefix):
    ini = f"""[circuitscape options]
scenario = pairwise
habitat_file = {os.path.abspath(res_asc)}
habitat_map_is_resistances = True
point_file = {os.path.abspath(node_asc)}
point_file_contains_polygons = True
output_file = {os.path.abspath(prefix + '.out')}
write_cur_maps = True
write_volt_maps = False
write_cum_cur_map_only = True
connect_four_neighbors_only = False
connect_using_avg_resistances = True
solver = cg+amg
print_timings = True
log_level = INFO
"""
    with open(prefix + '.ini', 'w') as f:
        f.write(ini)
    return prefix + '.ini'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--factor', type=int, default=3)
    ap.add_argument('--clip-to-study', action='store_true')
    ap.add_argument('--window', type=float, nargs=4)
    ap.add_argument('--label')
    ap.add_argument('--surface', default='', help='suffix of an alternative resistance raster, e.g. _negexp4')
    a = ap.parse_args()
    label = a.label or f'{a.model}_{30 * a.factor}m' + ('_studyclip' if a.clip_to_study else '') + a.surface
    out_dir = f'{C.OUT}/cs/{label}'
    os.makedirs(out_dir, exist_ok=True)
    res_asc, node_asc, t, shape, present = build_inputs(a.model, a.factor, a.clip_to_study, a.window, out_dir,
                                                        a.surface)
    print(f'{label}: grid {shape}, {shape[0] * shape[1] / 1e6:.2f} M cells, nodes {present}', flush=True)
    prefix = f'{out_dir}/{label}'
    ini = write_ini(res_asc, node_asc, prefix)
    t0 = time.time()
    with open(prefix + '_cs.log', 'w') as log:
        subprocess.run([CS_EXE, ini], check=True, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=BELOW_NORMAL)
    print(f'Circuitscape finished in {(time.time() - t0) / 60:.1f} min', flush=True)
    with rasterio.open(prefix + '_cum_curmap.asc') as s:
        cur = s.read(1).astype('float32')
    cur[cur < -9998] = np.nan
    with rasterio.open(f'{out_dir}/cum_current.tif', 'w', driver='GTiff', height=shape[0], width=shape[1],
                       count=1, dtype='float32', crs=C.CRS, transform=t, nodata=np.nan, compress='deflate') as d:
        d.write(cur, 1)
    print('saved', f'{out_dir}/cum_current.tif')


if __name__ == '__main__':
    main()
