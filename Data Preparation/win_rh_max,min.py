# -*- coding: utf-8 -*-
"""

@author: FatemehChajaei
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os

max_temp_ind_path = r"D:\Data\Urban Climate\Air Temp\Output\Daily_Maximum"
min_temp_ind_path = r"D:\Data\Urban Climate\Air Temp\Output\Daily_Minimum"


month_dict = {"01":"Jan", "05":"May", "07":"July", "10":"Oct"}
arr_shape = np.zeros((31, 301, 301))

max_ind = []
min_ind = []

for month_name in month_dict.values():
    
    # Load Maximum/Minimum of Temperature Index
    
    loaded_arr_temp_max = np.loadtxt(os.path.join(max_temp_ind_path, "max_temp_ind_" + month_name + ".txt"))
    
    max_ind.append(loaded_arr_temp_max.reshape(
        loaded_arr_temp_max.shape[0], loaded_arr_temp_max.shape[1] // arr_shape.shape[2], arr_shape.shape[2]))
    
    loaded_arr_temp_min = np.loadtxt(os.path.join(min_temp_ind_path, "min_temp_ind_" + month_name + ".txt"))
    
    min_ind.append(loaded_arr_temp_min.reshape(
        loaded_arr_temp_min.shape[0], loaded_arr_temp_min.shape[1] // arr_shape.shape[2], arr_shape.shape[2]))


""" Wind Speed """

wind_path = r"D:\Data\Urban Climate\Wind Speed"

m = 0
for month_num in month_dict.keys():
    fileName = "sfcWind_Amsterdam_UrbClim_2017_" + month_num + "_v1.0" + ".nc"
    file_location = os.path.join(wind_path, fileName)
    

    n1 = Dataset(file_location)
    var1 = n1.variables.keys()
    x1 = n1.variables['x']
    y1 = n1.variables['y']
    lat1 = n1.variables['latitude']
    lon1 = n1.variables['longitude']
    t1 = n1.variables['time']
    wind1 = n1.variables['sfcWind'][:]
    
    date_range = pd.period_range(start=str(t1.units[12:22]), periods=(t1.size)/24)
    
    max_wind_arr = np.zeros((date_range.size, y1.size, x1.size))
    min_wind_arr = np.zeros((date_range.size, y1.size, x1.size))
    
    for t in np.arange(0, date_range.size):
        for i in np.arange(0, y1.size):
            for j in np.arange(0, x1.size):
                max_wind_arr[t][i][j] = wind1[(t*24)+int(max_ind[m][t][i][j])][i][j]
     
    for t in np.arange(0, date_range.size):
        for i in np.arange(0, y1.size):
            for j in np.arange(0, x1.size):
                min_wind_arr[t][i][j] = wind1[(t*24)+int(min_ind[m][t][i][j])][i][j]
            
    m += 1

    # Save
    arr_reshaped_2 = min_wind_arr.reshape(min_wind_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(wind_path, "Output\Daily_Minimum" , "min_wind_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_2)
    
    arr_reshaped_3 = max_wind_arr.reshape(max_wind_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(wind_path, "Output\Daily_Maximum", "max_wind_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_3)
    
""" Relative Humidity """

rh_path = r"D:\Data\Urban Climate\Relative Humidity"

m = 0
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
    rh1 = n1.variables['russ'][:]
    
    date_range = pd.period_range(start=str(t1.units[12:22]), periods=(t1.size)/24)
    
    max_rh_arr = np.zeros((date_range.size, y1.size, x1.size))
    min_rh_arr = np.zeros((date_range.size, y1.size, x1.size))
    
    for t in np.arange(0, date_range.size):
        for i in np.arange(0, y1.size):
            for j in np.arange(0, x1.size):
                max_rh_arr[t][i][j] = rh1[(t*24)+int(max_ind[m][t][i][j])][i][j]
     
    for t in np.arange(0, date_range.size):
        for i in np.arange(0, y1.size):
            for j in np.arange(0, x1.size):
                min_rh_arr[t][i][j] = rh1[(t*24)+int(min_ind[m][t][i][j])][i][j]
            
    m += 1

    # Save
    arr_reshaped_2 = min_rh_arr.reshape(min_rh_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(rh_path, "Output\Daily_Minimum" , "min_rh_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_2)
    
    arr_reshaped_3 = max_rh_arr.reshape(max_rh_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(rh_path, "Output\Daily_Maximum", "max_rh_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_3)
 



