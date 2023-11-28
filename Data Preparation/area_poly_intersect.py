# -*- coding: utf-8 -*-
"""
@author: FatemehChajaei
"""

import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from osgeo import gdal
import numpy as np
import time
import os



bdg_footprint = r"D:\Data\Detected Building Footprints\CLIP_PARTS\bdg_clip_part14.shp"   # Detected Building Footprints
points_list_path = r"D:\Data\Amsterdam_XY_Points_5m.txt" # points within Amsterdam's study area captured at a 5-meter resolution
output_path = r"D:\Features\BDG\save_txt"

bdg_shp = GeoDataFrame.from_file(bdg_footprint)


points = []    
with open(points_list_path, "r") as f:
    for line in f:
        points.append(eval(line.strip()))
        

# buf_len = 27.5
buf_len = 52.5
# buf_len = 77.5
# buf_len = 102.5

# For faster runtime, we clip the building footprints and select the relevant index of points for each clipped part.

from alive_progress import alive_bar
with alive_bar(len(points[7856344:8383144])) as bar: 
    start_time = time.time()
    area_list = []
    for point in points[7856344:8383144:]:
        lowerleft = (point[0]-buf_len, point[1]-buf_len)
        lowerright = (point[0]+buf_len, point[1]-buf_len)
        topleft = (point[0]-buf_len, point[1]+buf_len)
        topright = (point[0]+buf_len, point[1]+buf_len)
    
        polys1 = geopandas.GeoSeries([Polygon([topleft, topright, lowerright, lowerleft])]) # create influence area around the measurment point
    
        df1 = geopandas.GeoDataFrame({'geometry': polys1})
    
    
        intersection_gdf = df1.overlay(bdg_shp, how='intersection')
    
        overlay_area = intersection_gdf['geometry'].area
        
        # ax = intersection_gdf.plot(cmap='tab10')
        # df1.plot(ax=ax, facecolor='none', edgecolor='k');
        
        area_list.append((np.sum(overlay_area))/df1['geometry'].area)
        bar()
        
        print("--- %s seconds ---" % (time.time() - start_time))


np.shape(area_list)

area_reshape = np.reshape(area_list, (150, 3512))  # base on the number of points in cliped part


np.savetxt(os.path.join(output_path, "area_intersect_50_clip_part14.txt"), area_reshape)

sample_raster_path = r"D:\Data\Amsterdam_DSM_Clip_5m.tif"  # the output raster should have same resolution and coordinate system as sample raster 

sample_ras = gdal.Open(sample_raster_path)

# Export the array as raster
A_raster = gdal.GetDriverByName('GTiff').Create(os.path.join(output_path, "Buffer", 'area_intersect_50.tif'), sample_ras.RasterXSize, sample_ras.RasterYSize, sample_ras.RasterCount, gdal.GDT_Float32)
A_raster.SetGeoTransform(sample_ras.GetGeoTransform())
A_raster.SetProjection(sample_ras.GetProjection())
A_raster.GetRasterBand(1).WriteArray(area_reshape)
A_raster.FlushCache()

