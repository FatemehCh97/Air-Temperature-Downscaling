# -*- coding: utf-8 -*-
"""
@author: FatemehChajaei
"""

import geopandas
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from osgeo import gdal
import numpy as np
import time
import os



bdg_footprint = r"D:\Data\Detected Building Footprints\CLIP_PARTS\bdg_clip_part10.shp"  # Detected Building Footprints (Clipped)
points_list_path =   r"D:\Data\Amsterdam_XY_Points_5m.txt" # points within Amsterdam's study area captured at a 5-meter resolution
bdg_attrubute_path = r"D:\Data\BDG_Attribute_Table_2.xls"
output_path = r"D:\Features\WALL\save_txt"
sample_raster_path =   r"D:\Data\Amsterdam_DSM_Clip_5m.tif"

bdg_shp = GeoDataFrame.from_file(bdg_footprint)


points = []    
with open(points_list_path, "r") as f:
    for line in f:
        points.append(eval(line.strip()))


# buf_len = 27.5
buf_len = 52.5
# buf_len = 77.5
# buf_len = 102.5


bdg_attribute_df = pd.read_excel(bdg_attrubute_path)
# bdg_attribute_df1 = bdg_attribute_df.drop(columns=['OBJECTID',  'gridcode', 'BUFF_DIST', 'ORIG_FID', 'ORIG_OID','STATUS', 'Area_calc', 'Shape_Leng',  'Id_1'])
bdg_attribute_df = bdg_attribute_df.set_index('Id')


# For faster runtime, we clip the building footprints and select the relevant index of points for each clipped part.
from alive_progress import alive_bar
with alive_bar(len(points[4600720:5478720])) as bar:
    start_time = time.time()
    wall_list = []
    for point in points[4600720:5478720]:
        lowerleft = (point[0]-buf_len, point[1]-buf_len)
        lowerright = (point[0]+buf_len, point[1]-buf_len)
        topleft = (point[0]-buf_len, point[1]+buf_len)
        topright = (point[0]+buf_len, point[1]+buf_len)
    
        polys1 = geopandas.GeoSeries([Polygon([topleft, topright, lowerright, lowerleft])])  # create influence area around the measurment point
    
        df1 = geopandas.GeoDataFrame({'geometry': polys1})
    
    
        intersection_gdf = df1.overlay(bdg_shp, how='intersection')
    
        overlay_area = intersection_gdf['geometry'].area
        intersection_gdf['overlay_area'] = overlay_area
        # overlay_bdg = intersection_gdf[['Id', 'overlay_area']]
        overlay_bdg = intersection_gdf[['Q_count', 'overlay_area']]
        
        weighted_sum_area = [(i[1]/bdg_attribute_df.loc[i[0]][0]) * bdg_attribute_df.loc[i[0]][1] for i in overlay_bdg.values]
        wall_list.append(np.sum(weighted_sum_area))
        
        bar()
    
        print("--- %s seconds ---" % (time.time() - start_time))


np.shape(wall_list)



# total_wall = np.reshape(wall_list, (2498, 3512))
total_wall = np.reshape(wall_list, (250, 3512)) # ((len(wall_list)/3512), 3512)

np.savetxt(os.path.join(output_path, "Wall_weights_buf50_clip_part11.txt"), total_wall)


sample_ras = gdal.Open(sample_raster_path)

# Export the array as raster
W_raster = gdal.GetDriverByName('GTiff').Create(os.path.join(output_path, 'wall_total_weighted_area_50.tif'), sample_ras.RasterXSize, sample_ras.RasterYSize, sample_ras.RasterCount, gdal.GDT_Float32)
W_raster.SetGeoTransform(sample_ras.GetGeoTransform())
W_raster.SetProjection(sample_ras.GetProjection())
W_raster.GetRasterBand(1).WriteArray(total_wall)
W_raster.FlushCache()

