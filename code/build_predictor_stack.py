"""Step 0: stack the eleven single-band predictor GeoTIFFs in data/predictors/ into the
11-band 30 m raster the analysis reads. The bands are stored separately because the
stacked file (about 450 MB) exceeds GitHub's 100 MB file limit."""
import os

import r1_config as C
import rasterio


def main():
    if os.path.exists(C.PREDICTOR_TIF):
        print('exists:', C.PREDICTOR_TIF)
        return
    files = [f'{C.DATA_DIR}/predictors/{i:02d}_{b}.tif' for i, b in enumerate(C.BAND_NAMES, 1)]
    with rasterio.open(files[0]) as s:
        prof = s.profile
    prof.update(count=len(files), compress='lzw', predictor=1, BIGTIFF='IF_SAFER')
    with rasterio.open(C.PREDICTOR_TIF, 'w', **prof) as dst:
        for i, (f, b) in enumerate(zip(files, C.BAND_NAMES), 1):
            with rasterio.open(f) as s:
                dst.write(s.read(1), i)
            dst.set_band_description(i, b)
            print('band', i, b, flush=True)
    print('written', C.PREDICTOR_TIF)


if __name__ == '__main__':
    main()
