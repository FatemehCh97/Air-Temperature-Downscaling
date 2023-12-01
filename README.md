# Air-Temperature-Downscaling

This repository contains the Python code used in the research paper titled "Machine Learning-Based Downscaling of Urban Air Temperature Using LiDAR Data" The code implements a framework based on machine learning techniques to downscale urban climate data obtained from the UrbClim model at a lower resolution (100 m) to a higher resolution (5 m).

## Data Preparation

The `Data Preparation` folder contains the code responsible for preparing the input data for the downscaling process. The urban climate data, including temperature, wind speed, and relative humidity, is provided in netCDF4 format and is available as hourly data for each month. The code in this folder extracts the necessary data and calculates the average, minimum, and maximum values of temperature, wind speed, and relative humidity for each day. These processed data are then used as input for the downscaling models.

## Building Footprint Detection

The `Building Footprint Detection` folder contains the code for extracting building footprints from lidar data. The extraction is performed using several deep semantic models, including U-Net, U-Net3+, Attention U-Net, and DeepLabV3+. The code in this folder implements the structure of these models and includes training and evaluation procedures. The best model is selected based on performance metrics, and transfer learning techniques are applied to fine-tune the model on Amsterdam data. Finally, the trained model is used to detect building footprints for the Amsterdam dataset.

## Machine Learning Models

The `Machine Learning Models` folder includes the code for training and testing various machine learning algorithms for estimating temperatures. The code prepares the training and test datasets and implements algorithms such as xgboost, extra trees, lightgbm, random forest, and svr. These models utilize morphological and meteorological data, to predict temperature values.

Please refer to the accompanying research paper for more detailed information regarding the methodology, experimental setup, and results.
