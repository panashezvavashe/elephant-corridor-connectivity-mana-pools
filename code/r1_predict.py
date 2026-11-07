"""Step 3: predict the revised RF and SVM models across the 30 m predictor grid.

Writes, per model: probability (P), resistance = max(1 - P, 0.001) (manuscript §2.7),
and the binary corridor at the nested MaxTSS threshold from r1_models.py (#16).
The raster is processed in strips of rows so the full 11-band stack never sits in
memory at once, which also keeps the CPU load (and laptop temperature) moderate.
"""
import json
import sys
import time

import r1_config as C
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

STRIP_ROWS = 150
MIN_RESISTANCE = 0.001


def features(block):
    """block: (bands, rows, cols) -> DataFrame in model feature order + valid mask."""
    b, r, c = block.shape
    df = pd.DataFrame(block.reshape(b, r * c).T, columns=C.BAND_NAMES)
    valid = ~((df == -9999).any(axis=1) | df.isna().any(axis=1)).values
    for k in C.LULC_CLASSES:
        df[f'LULC_{k}'] = (df['LandCover'] == k).astype(float)
    return df.loc[valid, C.FEATURES].astype(float), valid


def main(models=('RF', 'SVM')):
    settings = json.load(open(C.OUT_DATA + '/model_settings.json'))
    thr = settings['thresholds']
    fitted = {m: joblib.load(f'{C.OUT_DATA}/{m.lower()}_model.joblib') for m in models}
    with rasterio.open(C.PREDICTOR_TIF) as src:
        meta = src.meta.copy()
        meta.update(count=1, dtype='float32', nodata=np.nan, compress='deflate', tiled=True,
                    blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
        H, W = src.height, src.width
        outs = {}
        for m in models:
            for kind in ('probability', 'resistance', 'binary_corridor'):
                outs[(m, kind)] = rasterio.open(f'{C.OUT_RAS}/{m.lower()}_{kind}.tif', 'w', **meta)
        t0 = time.time()
        for r0 in range(0, H, STRIP_ROWS):
            n = min(STRIP_ROWS, H - r0)
            win = Window(0, r0, W, n)
            X, valid = features(src.read(window=win))
            for m in models:
                prob = np.full(n * W, np.nan, dtype='float32')
                if len(X):
                    prob[valid] = fitted[m].predict_proba(X)[:, 1]
                res = np.where(np.isnan(prob), np.nan, np.maximum(1 - prob, MIN_RESISTANCE)).astype('float32')
                binr = np.where(np.isnan(prob), np.nan, (prob >= thr[m]).astype('float32')).astype('float32')
                for kind, arr in (('probability', prob), ('resistance', res), ('binary_corridor', binr)):
                    outs[(m, kind)].write(arr.reshape(n, W), 1, window=win)
            done = (r0 + n) / H
            el = (time.time() - t0) / 60
            print(f'{done:6.1%}  {el:6.1f} min elapsed, ~{el / done - el:6.1f} min left', flush=True)
        for ds in outs.values():
            ds.close()
    print('prediction done')


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or ('RF', 'SVM'))
