# -*- coding: utf-8 -*-
"""

@author: FatemehChajaei
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os

path_test = r"D:\Data\Urban Climate\Air Temp\tas_Amsterdam_UrbClim_2017_01_v1.0.nc"

# month = ["01", "05", "07", "10"]
month_dict = {"01":"Jan", "05":"May", "07":"July", "10":"Oct"}

"""Air Temperature"""

temp_path = r"D:\Data\Urban Climate\Air Temp"
temp_out_path = r"D:\Data\Urban Climate\Air Temp\Output\Daily_Average"

for month_num in month_dict.keys():
    fileName = "tas_Amsterdam_UrbClim_2017_" + month_num + "_v1.0" + ".nc"
    file_location = os.path.join(temp_path, fileName)
    

    n1 = Dataset(file_location)
    var1 = n1.variables.keys()
    x1 = n1.variables['x']
    y1 = n1.variables['y']
    lat1 = n1.variables['latitude']
    lon1 = n1.variables['longitude']
    t1 = n1.variables['time']
    temp1 = n1.variables['tas']
    
    date_range = pd.period_range(start=str(t1.units[12:22]), periods=(t1.size)/24)
    
    avg_temp_arr = np.zeros((date_range.size, y1.size, x1.size))
    avg_temp_arr_3sigma = np.zeros((date_range.size, y1.size, x1.size))


    # Calculating mean
    for t in np.arange(0, date_range.size):
                # avg_temp_arr[t, i, j] = (np.sum(temp1[(t*24):(t*24)+24, i, j]))/24
                avg_temp_arr[t] = np.mean(temp1[(t*24):(t*24)+24], axis = 0)
    
    # average daily temperature with applying 3 sigam test
    for t in np.arange(0, date_range.size):
        sigma = np.std(temp1[(t*24):(t*24)+24], axis=0)
        a = np.subtract(avg_temp_arr[t], 3 * sigma)
        b = np.add(avg_temp_arr[t], 3 * sigma)
        
        for i in np.arange(0, y1.size):
            for j in np.arange(0, x1.size):      
                day_ij_avg = [x for x in temp1[(t*24):(t*24)+24, i, j] if x > a[i,j] or x < b[i,j] ]
        
                avg_temp_arr_3sigma[t, i, j] = np.mean(day_ij_avg)
    

    # Save the output in 2D text format, reshape to original shape after loading
    arr_reshaped = avg_temp_arr_3sigma.reshape(avg_temp_arr_3sigma.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(temp_out_path, "avg_temp_arr_" + month_dict[month_num] + "_3sigma.txt"), arr_reshaped)
     
    arr_reshaped_2 = avg_temp_arr.reshape(avg_temp_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(temp_out_path, "avg_temp_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_2)
  

"""Relative Humidity"""

rh_path = r"D:\Data\Urban Climate\Relative Humidity"
rh_out_path = r"D:\Data\Urban Climate\Relative Humidity\Output\Daily_Average"

for month_num in month_dict.keys():
    # fileName = "tas_Amsterdam_UrbClim_2017_" + month_num + "_v1.0" + ".nc"
    fileName = "russ_Amsterdam_UrbClim_2017_" + month_num + "_v1.0" + ".nc"
    file_location = os.path.join(rh_path, fileName)
    

    n1 = Dataset(file_location)
    var1 = n1.variables.keys()
    x1 = n1.variables['x']
    y1 = n1.variables['y']
    lat1 = n1.variables['latitude']
    lon1 = n1.variables['longitude']
    t1 = n1.variables['time']
    rh1 = n1.variables['russ']
    
    date_range = pd.period_range(start=str(t1.units[12:22]), periods=(t1.size)/24)
    
    avg_rh_arr = np.zeros((date_range.size, y1.size, x1.size))
    avg_rh_arr_3sigma = np.zeros((date_range.size, y1.size, x1.size))


    # Calculating mean
    for t in np.arange(0, date_range.size):
                avg_rh_arr[t] = np.mean(rh1[(t*24):(t*24)+24], axis = 0)
    
    # average daily temperature with applying 3 sigam test
    for t in np.arange(0, date_range.size):
        sigma = np.std(rh1[(t*24):(t*24)+24], axis=0)
        a = np.subtract(avg_rh_arr[t], 3 * sigma)
        b = np.add(avg_rh_arr[t], 3 * sigma)
        
        for i in np.arange(0, y1.size):
            for j in np.arange(0, x1.size):      
                day_ij_avg = [x for x in rh1[(t*24):(t*24)+24, i, j] if x > a[i,j] or x < b[i,j] ]
        
                avg_rh_arr_3sigma[t, i, j] = np.mean(day_ij_avg)
    

    # Save the output in 2D text format, reshape to original shape after loading
    arr_reshaped = avg_rh_arr_3sigma.reshape(avg_rh_arr_3sigma.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(rh_out_path, "avg_rh_arr_" + month_dict[month_num] + "_3sigma.txt"), arr_reshaped)
     
    arr_reshaped_2 = avg_rh_arr.reshape(avg_rh_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(rh_out_path, "avg_rh_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_2)
  

"""Wind Speed"""

wind_path = r"D:\Data\Urban Climate\Wind Speed"
wind_out_path = r"D:\Data\Urban Climate\Wind Speed\Output\Daily_Average"

for month_num in month_dict.keys():
    # fileName = "tas_Amsterdam_UrbClim_2017_" + month_num + "_v1.0" + ".nc"
    fileName = "sfcWind_Amsterdam_UrbClim_2017_" + month_num + "_v1.0" + ".nc"
    file_location = os.path.join(wind_path, fileName)
    

    n1 = Dataset(file_location)
    var1 = n1.variables.keys()
    x1 = n1.variables['x']
    y1 = n1.variables['y']
    lat1 = n1.variables['latitude']
    lon1 = n1.variables['longitude']
    t1 = n1.variables['time']
    wind1 = n1.variables['sfcWind']
    
    date_range = pd.period_range(start=str(t1.units[12:22]), periods=(t1.size)/24)
    
    avg_wind_arr = np.zeros((date_range.size, y1.size, x1.size))
    avg_wind_arr_3sigma = np.zeros((date_range.size, y1.size, x1.size))


    # Calculating mean
    for t in np.arange(0, date_range.size):
                avg_wind_arr[t] = np.mean(wind1[(t*24):(t*24)+24], axis = 0)
    
    # average daily temperature with applying 3 sigam test
    for t in np.arange(0, date_range.size):
        sigma = np.std(wind1[(t*24):(t*24)+24], axis=0)
        a = np.subtract(avg_wind_arr[t], 3 * sigma)
        b = np.add(avg_wind_arr[t], 3 * sigma)
        
        for i in np.arange(0, y1.size):
            for j in np.arange(0, x1.size):      
                day_ij_avg = [x for x in wind1[(t*24):(t*24)+24, i, j] if x > a[i,j] or x < b[i,j] ]
        
                avg_wind_arr_3sigma[t, i, j] = np.mean(day_ij_avg)
    

    # Save the output in 2D text format, reshape to original shape after loading
    arr_reshaped = avg_wind_arr_3sigma.reshape(avg_wind_arr_3sigma.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(wind_out_path, "avg_wind_arr_" + month_dict[month_num] + "_3sigma.txt"), arr_reshaped)
     
    arr_reshaped_2 = avg_wind_arr.reshape(avg_wind_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(wind_out_path, "avg_wind_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_2)
    
    
## Loading the saved file
# for month_name in month_dict.values():
#     loaded_arr = np.loadtxt(os.path.join(out_path, "avg_temp_arr_" + month_name + "_3sigma.txt"))
    
#     avg_temp_load = loaded_arr.reshape(
#         loaded_arr.shape[0], loaded_arr.shape[1] // avg_temp_arr_3sigma.shape[2], avg_temp_arr_3sigma.shape[2])
    
#     for day in np.arange(0, date_range.size):
#     avg_temp_raster = arcpy.NumPyArrayToRaster(avg_temp_load[day], lower_left_corner=lower_left_corner, x_cell_size=100)
#     avg_temp_raster.save(os.path.join(save_dir, "Temp_avg_" + month_name + ".gdb", "Temp_"+ f"{day:02d}"))
      
    