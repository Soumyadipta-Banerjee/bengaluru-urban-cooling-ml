"""
Assignment 4: Urban Cooling Analysis - Earth Engine Python API Script
Automated extraction of AlphaEarth and Landsat LST data.
"""

import os
import sys
import ee

def init_earth_engine():
    try:
        ee.Initialize()
        print("Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        print("\nTo authenticate, please run in your terminal:")
        print("    earthengine authenticate")
        print("and follow the Google login instructions.")
        return False

def main():
    if not init_earth_engine():
        sys.exit(1)

    # Bengaluru metropolitan region bounding box
    bengaluru_bbox = ee.Geometry.BBox(77.45, 12.83, 77.79, 13.15)
    studyArea = bengaluru_bbox
    year = '2024'
    startDate = f'{year}-01-01'
    endDate = f'{year}-12-31'

    # 1. Landsat 8 LST
    def mask_l8(image):
        qa = image.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1 << 4).eq(0).And(qa.bitwiseAnd(1 << 3).eq(0))
        lst = image.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
        return image.addBands(lst).updateMask(mask)

    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(studyArea) \
        .filterDate(startDate, endDate) \
        .map(mask_l8)
    lst_median = landsat.select('LST').median().clip(studyArea)

    # 2. Dynamic World
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterBounds(studyArea) \
        .filterDate(startDate, endDate).mean().clip(studyArea)
    
    green_mask = dw.select('trees').add(dw.select('grass')).gte(0.60)
    built_mask = dw.select('built').gte(0.60)
    water_mask = dw.select('water').gte(0.30)

    # 3. Cooling intensity calculation
    ring500 = green_mask.focal_max(500, 'circle', 'meters')
    ring200 = green_mask.focal_max(200, 'circle', 'meters')
    ring_zone = ring500.subtract(ring200).gt(0)
    urban_ref_ring = ring_zone.And(built_mask).And(water_mask.Not()).And(green_mask.Not())

    urban_ring_lst = lst_median.updateMask(urban_ref_ring)
    ring_median_lst = urban_ring_lst.reduceNeighborhood(
        reducer=ee.Reducer.median(),
        kernel=ee.Kernel.circle(500, 'meters')
    ).rename('LST_ring_median')

    green_lst = lst_median.updateMask(green_mask).rename('LST_green')
    cooling_intensity = ring_median_lst.subtract(green_lst).rename('cooling_intensity')

    # 4. AlphaEarth
    bands_64 = [f"A{i:02d}" for i in range(64)]
    alpha = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING_V1_ANNUAL') \
        .filterBounds(studyArea) \
        .filterDate(startDate, endDate)
    alpha_img = ee.Image(ee.Algorithms.If(
        alpha.size().gt(0),
        alpha.first(),
        ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING_V1_ANNUAL').filterBounds(studyArea).sort('system:time_start', False).first()
    )).clip(studyArea).select(bands_64)

    # 5. Sampling
    full_image = cooling_intensity.addBands(green_lst).addBands(ring_median_lst).addBands(alpha_img)
    samples = full_image.sample(region=studyArea, scale=30, numPixels=3500, geometries=True, seed=42)

    def add_coords(feat):
        coords = feat.geometry().coordinates()
        return feat.set({'longitude': coords.get(0), 'latitude': coords.get(1)})

    samples = samples.map(add_coords)

    # Export tasks to Drive
    task1 = ee.batch.Export.table.toDrive(
        collection=samples,
        description='bengaluru_cooling_samples_2024',
        folder='GEE_Urban_Cooling',
        fileFormat='CSV'
    )
    task1.start()
    print("Started Export Table task (check Earth Engine Tasks dashboard).")

if __name__ == '__main__':
    main()
