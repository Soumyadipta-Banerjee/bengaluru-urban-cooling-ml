/**
 * Assignment 4: Urban Cooling Island (UCI) Analysis in Bengaluru
 * Google Earth Engine Script for Data Extraction & Spatial Preprocessing
 * 
 * Instructions:
 * 1. Open Google Earth Engine Code Editor: https://code.earthengine.google.com/
 * 2. Paste this script into the editor.
 * 3. Click "Run". You will see map visualization layers.
 * 4. Go to the "Tasks" tab on the right panel and click "Run" on both export tasks:
 *    - Export Table: bengaluru_cooling_samples_2024 (CSV)
 *    - Export Image: bengaluru_alphaearth_30m_utm (GeoTIFF)
 * 5. Download the files from Google Drive to your local project folder `data/raw/`.
 */

// =========================================================================
// 1. Study Area: Bengaluru Metropolitan Region (BBMP + Outer Buffer)
// =========================================================================
// Bounding box for Bengaluru metropolitan region [minLon, minLat, maxLon, maxLat]
var bengaluru_bbox = ee.Geometry.BBox(77.45, 12.83, 77.79, 13.15);

// If you have uploaded the BBMP shapefile/GeoJSON as an EE asset, you can replace this:
var studyArea = bengaluru_bbox;
Map.centerObject(studyArea, 11);

var year = '2024';
var startDate = year + '-01-01';
var endDate = year + '-12-31';

// =========================================================================
// 2. Landsat 8 Collection 2 Level-2 Surface Temperature (LST)
// =========================================================================
function maskLandsat8Sr(image) {
  var qa = image.select('QA_PIXEL');
  // Bit 3 = Cloud, Bit 4 = Cloud Shadow, Bit 1 = Dilated Cloud
  var cloudShadowMask = qa.bitwiseAnd(1 << 4).eq(0);
  var cloudsMask = qa.bitwiseAnd(1 << 3).eq(0);
  var dilatedCloudMask = qa.bitwiseAnd(1 << 1).eq(0);
  var mask = cloudShadowMask.and(cloudsMask).and(dilatedCloudMask);
  
  // ST_B10 scale factor: 0.00341802, offset: 149.0 (Kelvin) -> subtract 273.15 for Celsius
  var lstCelsius = image.select('ST_B10')
    .multiply(0.00341802)
    .add(149.0)
    .subtract(273.15)
    .rename('LST');
    
  return image.addBands(lstCelsius).updateMask(mask);
}

var landsatCollection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(studyArea)
  .filterDate(startDate, endDate)
  .map(maskLandsat8Sr);

var lstMedian = landsatCollection.select('LST').median().clip(studyArea);

// =========================================================================
// 3. Dynamic World V1 (10 m Land-cover Probabilities)
// =========================================================================
var dwCollection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')
  .filterBounds(studyArea)
  .filterDate(startDate, endDate);

var dwMean = dwCollection.mean().clip(studyArea);

// Dynamic World classes: trees, grass, built, water
var treesProb = dwMean.select('trees');
var grassProb = dwMean.select('grass');
var builtProb = dwMean.select('built');
var waterProb = dwMean.select('water');

// Thresholds as per specification (probability >= 0.60)
var greenMask = treesProb.add(grassProb).gte(0.60).rename('green_mask');
var builtMask = builtProb.gte(0.60).rename('built_mask');
var waterMask = waterProb.gte(0.30).rename('water_mask');

// =========================================================================
// 4. Urban Reference Ring (200 m - 500 m) & Cooling Intensity (ΔT)
// =========================================================================
// Reference ring surrounding green spaces: 200m to 500m
// Built-up pixels only, excluding water bodies and other green patches
var ring500 = greenMask.focal_max({radius: 500, units: 'meters'});
var ring200 = greenMask.focal_max({radius: 200, units: 'meters'});
var ringZone = ring500.subtract(ring200).gt(0);

// Final urban reference ring: ring zone AND built-up AND NOT water AND NOT green
var urbanRefRing = ringZone.and(builtMask).and(waterMask.not()).and(greenMask.not());

// Masked LST in urban ring
var urbanRingLST = lstMedian.updateMask(urbanRefRing);

// Compute local median LST in 500m neighborhood for the reference ring
var ringMedianLST = urbanRingLST.reduceNeighborhood({
  reducer: ee.Reducer.median(),
  kernel: ee.Kernel.circle({radius: 500, units: 'meters'})
}).rename('LST_ring_median');

// LST of green core
var greenLST = lstMedian.updateMask(greenMask).rename('LST_green');

// Cooling intensity: median LST of surrounding built ring minus LST of green core
var coolingIntensity = ringMedianLST.subtract(greenLST).rename('cooling_intensity');

// =========================================================================
// 5. AlphaEarth Satellite Embedding V1 (64 bands: A00 - A63)
// =========================================================================
var alphaEarthCol = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING_V1_ANNUAL')
  .filterBounds(studyArea)
  .filterDate(startDate, endDate);

// If 2024 is pending in standard catalog, fall back to the latest available year
var alphaEarth = ee.Algorithms.If(
  alphaEarthCol.size().gt(0),
  alphaEarthCol.first(),
  ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING_V1_ANNUAL').filterBounds(studyArea).sort('system:time_start', false).first()
);
alphaEarth = ee.Image(alphaEarth).clip(studyArea);

// Select bands A00 to A63
var embeddingBands = [];
for (var i = 0; i < 64; i++) {
  var bName = 'A' + (i < 10 ? '0' + i : '' + i);
  embeddingBands.push(bName);
}
var alphaEarth64 = alphaEarth.select(embeddingBands);

// =========================================================================
// 6. Combine Features & Extract Sample Points for Model Training
// =========================================================================
// Combine cooling intensity and AlphaEarth bands
var fullDatasetImage = coolingIntensity
  .addBands(greenLST)
  .addBands(ringMedianLST)
  .addBands(alphaEarth64);

// Sample stratified points on green locations (removing NoData)
var samplePoints = fullDatasetImage.sample({
  region: studyArea,
  scale: 30, // 30 m grid as required
  numPixels: 3500, // Sufficient spatial sample size
  geometries: true,
  seed: 42
});

// Add longitude and latitude for spatial stratified splitting
samplePoints = samplePoints.map(function(feature) {
  var coords = feature.geometry().coordinates();
  return feature.set({
    'longitude': coords.get(0),
    'latitude': coords.get(1)
  });
});

// =========================================================================
// 7. Map Visualizations
// =========================================================================
Map.addLayer(lstMedian, {min: 22, max: 40, palette: ['blue', 'cyan', 'green', 'yellow', 'red']}, 'Landsat 8 LST (°C)');
Map.addLayer(greenMask.updateMask(greenMask), {palette: ['00FF00']}, 'Confident Green Spaces (>0.60)');
Map.addLayer(coolingIntensity, {min: 0, max: 8, palette: ['white', 'yellow', 'blue']}, 'Cooling Intensity (ΔT °C)');
Map.addLayer(samplePoints, {color: 'FF00FF'}, 'Sample Points (k-NN dataset)');

// =========================================================================
// 8. Export Tasks (Run these from the Tasks tab on the right)
// =========================================================================

// Export 1: Tabular Dataset for Model Training (CSV)
Export.table.toDrive({
  collection: samplePoints,
  description: 'bengaluru_cooling_samples_2024',
  folder: 'GEE_Urban_Cooling',
  fileFormat: 'CSV'
});

// Export 2: AlphaEarth 64-band Stack at 30m in EPSG:32643 (UTM Zone 43N)
Export.image.toDrive({
  image: alphaEarth64,
  description: 'bengaluru_alphaearth_30m_utm',
  folder: 'GEE_Urban_Cooling',
  scale: 30,
  crs: 'EPSG:32643',
  region: studyArea,
  maxPixels: 1e9
});

print('Workflow script initialized. Check Tasks tab to export table and image.');
