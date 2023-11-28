# -*- coding: utf-8 -*-
"""

@author: Fatemeh Chajaei
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os

path_test = r"D:\Data\Urban Climate\Air Temp\tas_Amsterdam_UrbClim_2017_01_v1.0.nc"

# month = ["01", "05", "07", "10"]
month_dict = {"01":"Jan", "05":"May", "07":"July", "10":"Oct"}

"""Minimum Air Temperature"""

temp_path = r"D:\Data\Urban Climate\Air Temp"
temp_out_path = r"D:\Data\Urban Climate\Air Temp\Output\Daily_Minimum"

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
    
    # avg_temp_arr = np.zeros((date_range.size, y1.size, x1.size))
    min_temp_arr = np.zeros((date_range.size, y1.size, x1.size))
    min_temp_ind = np.zeros((date_range.size, y1.size, x1.size))
    # min_temp_arr_3sigma = np.zeros((date_range.size, y1.size, x1.size))


    # Calculating mean for 3sigma test
    for t in np.arange(0, date_range.size):
        # avg_temp_arr[t] = np.mean(temp1[(t*24):(t*24)+24], axis = 0)
        min_temp_arr[t] = np.min(temp1[(t*24):(t*24)+24], axis = 0)
        min_temp_ind[t] = np.argmin(temp1[(t*24):(t*24)+24], axis = 0)
    
    # daily temperature with applying 3 sigam test
    # for t in np.arange(0, date_range.size):
    #     sigma = np.std(temp1[(t*24):(t*24)+24], axis=0)
    #     a = np.subtract(avg_temp_arr[t], 3 * sigma)
    #     b = np.add(avg_temp_arr[t], 3 * sigma)
        
    #     for i in np.arange(0, y1.size):
    #         for j in np.arange(0, x1.size):      
    #             day_ij_lim = [x for x in temp1[(t*24):(t*24)+24, i, j] if x > a[i,j] or x < b[i,j]]
        
    #             min_temp_arr_3sigma[t, i, j] = np.min(day_ij_lim)
    

    # Save the output in 2D text format, reshape to original shape after loading
    # arr_reshaped = min_temp_arr_3sigma.reshape(min_temp_arr_3sigma.shape[0], -1)
      
    # # saving reshaped array to file.
    # np.savetxt(os.path.join(temp_out_path, "max_temp_arr_" + month_dict[month_num] + "_3sigma.txt"), arr_reshaped)
     
    arr_reshaped_2 = min_temp_arr.reshape(min_temp_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(temp_out_path, "min_temp_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_2)
    
    arr_reshaped_3 = min_temp_ind.reshape(min_temp_ind.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(temp_out_path, "min_temp_ind_" + month_dict[month_num] + ".txt"), arr_reshaped_3)
  

""" Maximum  Air Temperature"""

temp_out_path = r"D:\Data\Urban Climate\Air Temp\Output\Daily_Maximum"

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
    
    # avg_temp_arr = np.zeros((date_range.size, y1.size, x1.size))
    max_temp_arr = np.zeros((date_range.size, y1.size, x1.size))
    max_temp_ind = np.zeros((date_range.size, y1.size, x1.size))
    # max_temp_arr_3sigma = np.zeros((date_range.size, y1.size, x1.size))


    # Calculating mean for 3sigma test and mean, max
    for t in np.arange(0, date_range.size):
        # avg_temp_arr[t] = np.mean(temp1[(t*24):(t*24)+24], axis = 0)
        max_temp_arr[t] = np.max(temp1[(t*24):(t*24)+24], axis = 0)
        max_temp_ind[t] = np.argmax(temp1[(t*24):(t*24)+24], axis = 0)
    
    # average daily temperature with applying 3 sigam test
    # for t in np.arange(0, date_range.size):
    #     sigma = np.std(temp1[(t*24):(t*24)+24], axis=0)
    #     a = np.subtract(avg_temp_arr[t], 3 * sigma)
    #     b = np.add(avg_temp_arr[t], 3 * sigma)
        
    #     for i in np.arange(0, y1.size):
    #         for j in np.arange(0, x1.size):      
    #             day_ij_lim = [x for x in temp1[(t*24):(t*24)+24, i, j] if x > a[i,j] or x < b[i,j]]
        
    #             max_temp_arr_3sigma[t, i, j] = np.max(day_ij_lim)
    

    # Save the output in 2D text format, reshape to original shape after loading
    # arr_reshaped = max_temp_arr_3sigma.reshape(max_temp_arr_3sigma.shape[0], -1)
      
    # # saving reshaped array to file.
    # np.savetxt(os.path.join(temp_out_path, "max_temp_arr_" + month_dict[month_num] + "_3sigma.txt"), arr_reshaped)
     
    arr_reshaped_2 = max_temp_arr.reshape(max_temp_arr.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(temp_out_path, "max_temp_arr_" + month_dict[month_num] + ".txt"), arr_reshaped_2)
    
    arr_reshaped_3 = max_temp_ind.reshape(max_temp_ind.shape[0], -1)
      
    # saving reshaped array to file.
    np.savetxt(os.path.join(temp_out_path, "max_temp_ind_" + month_dict[month_num] + ".txt"), arr_reshaped_3)
  