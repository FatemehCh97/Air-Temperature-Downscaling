# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 02:19:19 2023

@author: NovinGostar
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

df =pd.read_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\UrbClim_Prediction_Temps_Points_2.csv")

# df = df.drop(['OID_'], axis=1)

# for i in range(0,12):
#     df = df.rename(columns={df.columns[i]: df.columns[i][8:].replace("_"," ")})

# for i in range(12,24):
#     df = df.rename(columns={df.columns[i]: (df.columns[i][:8] + df.columns[i][17:]).replace("_"," ")})


# df = df[['Prediction Jan 17', 'UrbClim Jan 17', 'Prediction Jan 18', 'UrbClim Jan 18', 'Prediction Jan 19', 'UrbClim Jan 19',
#           'Prediction May 04', 'UrbClim May 04', 'Prediction May 05', 'UrbClim May 05', 'Prediction May 06', 'UrbClim May 06',
#           'Prediction July 18', 'UrbClim July 18', 'Prediction July 19', 'UrbClim July 19', 'Prediction July 20', 'UrbClim July 20',
#           'Prediction Oct 24', 'UrbClim Oct 24', 'Prediction Oct 25', 'UrbClim Oct 25', 'Prediction Oct 26', 'UrbClim Oct 26']]

# df.to_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\UrbClim_Prediction_Temps_Points_2.csv" ,index=False)


# Separate "Prediction" and "UrbClim" columns
prediction_cols = [col for col in df.columns if col.startswith('Prediction')]
urbclim_cols = [col for col in df.columns if col.startswith('UrbClim')]

columns = df.columns

labels_col = [col.replace('Prediction ', '').replace('UrbClim ', '') for col in columns]
labels_col_1 = list(dict.fromkeys(labels_col))

# Create box plot
fig, ax = plt.subplots(figsize=(12, 7))

boxprops_pred = dict(edgecolor='teal', facecolor='lightseagreen')
medianprops_pred = dict(color='teal')
whiskerprops_pred = dict(color='teal')
capprops_pred = dict(color='teal')

boxprops_urb = dict(edgecolor='g', facecolor='mediumseagreen')
medianprops_urb = dict(color='g')
whiskerprops_urb = dict(color='g')
capprops_urb = dict(color='g')

positions = np.arange(1, len(columns) + 1)

handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor='lightseagreen', edgecolor='teal'),
           mpatches.Rectangle((0, 0), 1, 1, facecolor='mediumseagreen', edgecolor='g')]
labels = ['Prediction', 'UrbClim']


for i, col in enumerate(columns):
    if i % 2 == 0:
        ax.boxplot(df[col], positions=[positions[i]], patch_artist=True, boxprops=boxprops_pred, whiskerprops=whiskerprops_pred,
                   capprops=capprops_pred, medianprops=medianprops_pred, widths=0.6, showfliers=False)
    else:
        ax.boxplot(df[col], positions=[positions[i]], patch_artist=True, boxprops=boxprops_urb, whiskerprops=whiskerprops_urb,
                   capprops=capprops_urb, medianprops=medianprops_urb, widths=0.6, showfliers=False)

# Set x-axis labels
ax.set_xticks(positions)
ax.set_xticklabels(labels_col, rotation=45, fontname='Times New Roman', fontsize=12)
# ax.set_xticks(positions[::2]+0.5)
# ax.set_xticklabels(labels_col_1, fontname='Times New Roman', fontsize=14)
# ax.set_xticklabels(columns, rotation=90, fontname='Times New Roman', fontsize=10)

# Set the y-axis label
# ax.set_yticklabels(fontname='Times New Roman', fontsize=14)
ax.set_ylabel('Temperature (K)', labelpad=10, fontname='Times New Roman', fontsize=14)

# Add legend
ax.legend(handles, labels, loc='upper right', frameon=False)

# Save Fig
plt.savefig(r'D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\Results\Box_Plot_1.png', bbox_inches='tight', dpi=350)

# Display the plot
plt.show()


# Dictionary to store column properties
column_properties = {}

# Calculate statistics for each column (box plot)
for column in df.columns:
    data = df[column]
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    # outliers = [x for x in data if x < lower_fence or x > upper_fence]
    
    # Store the statistics in the dictionary
    column_properties[column] = {
        "Median": median,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence,
        # "Outliers": outliers
    }

# Create a DataFrame from the dictionary
column_properties_df = pd.DataFrame.from_dict(column_properties, orient='index')


prediction_df = column_properties_df[column_properties_df.index.str.startswith('Prediction')]
urbclim_df = column_properties_df[column_properties_df.index.str.startswith('UrbClim')]

# Get the column names to iterate over
column_names = column_properties_df.columns

# Create an empty DataFrame to store the differences
difference_df = pd.DataFrame(index=prediction_df.index)

# Calculate the difference between the Median values of each corresponding row
for column in column_names:
    difference_df[column + ' Difference'] = prediction_df[column].values - urbclim_df[column].values

# Display the resulting DataFrame
print(difference_df)



"""MAX Temp"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

df =pd.read_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\MAX\UrbClim_Prediction_Temps_Points.csv")

# df = df.drop(['OID_', 'pointid'], axis=1)

# for i in range(0,12):
#     df = df.rename(columns={df.columns[i]: df.columns[i][8:].replace("_"," ")})

# for i in range(12,24):
#     df = df.rename(columns={df.columns[i]: (df.columns[i][:8] + df.columns[i][17:]).replace("_"," ")})


# df = df[['Prediction Jan 17', 'UrbClim Jan 17', 'Prediction Jan 18', 'UrbClim Jan 18', 'Prediction Jan 19', 'UrbClim Jan 19',
#           'Prediction May 04', 'UrbClim May 04', 'Prediction May 05', 'UrbClim May 05', 'Prediction May 06', 'UrbClim May 06',
#           'Prediction July 18', 'UrbClim July 18', 'Prediction July 19', 'UrbClim July 19', 'Prediction July 20', 'UrbClim July 20',
#           'Prediction Oct 24', 'UrbClim Oct 24', 'Prediction Oct 25', 'UrbClim Oct 25', 'Prediction Oct 26', 'UrbClim Oct 26']]

# df.to_csv(r"D:\UNI\M\Thesis\2\SFW_Code\#Code\#Temp_Prediction_Amsterdam\Preds\MAX\UrbClim_Prediction_Temps_Points_2.csv" ,index=False)


# Separate "Prediction" and "UrbClim" columns
prediction_cols = [col for col in df.columns if col.startswith('Prediction')]
urbclim_cols = [col for col in df.columns if col.startswith('UrbClim')]

columns = df.columns

labels_col = [col.replace('Prediction ', '').replace('UrbClim ', '') for col in columns]
labels_col_1 = list(dict.fromkeys(labels_col))

# Create box plot
fig, ax = plt.subplots(figsize=(12, 7))

boxprops_pred = dict(edgecolor='teal', facecolor='lightseagreen')
medianprops_pred = dict(color='teal')
whiskerprops_pred = dict(color='teal')
capprops_pred = dict(color='teal')

boxprops_urb = dict(edgecolor='g', facecolor='mediumseagreen')
medianprops_urb = dict(color='g')
whiskerprops_urb = dict(color='g')
capprops_urb = dict(color='g')

positions = np.arange(1, len(columns) + 1)

handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor='lightseagreen', edgecolor='teal'),
           mpatches.Rectangle((0, 0), 1, 1, facecolor='mediumseagreen', edgecolor='g')]
labels = ['Prediction', 'UrbClim']


for i, col in enumerate(columns):
    if i % 2 == 0:
        ax.boxplot(df[col], positions=[positions[i]], patch_artist=True, boxprops=boxprops_pred, whiskerprops=whiskerprops_pred,
                   capprops=capprops_pred, medianprops=medianprops_pred, widths=0.6, showfliers=False)
    else:
        ax.boxplot(df[col], positions=[positions[i]], patch_artist=True, boxprops=boxprops_urb, whiskerprops=whiskerprops_urb,
                   capprops=capprops_urb, medianprops=medianprops_urb, widths=0.6, showfliers=False)

# Set x-axis labels
ax.set_xticks(positions)
ax.set_xticklabels(labels_col, rotation=45, fontname='Times New Roman', fontsize=12)
# ax.set_xticks(positions[::2]+0.5)
# ax.set_xticklabels(labels_col_1, fontname='Times New Roman', fontsize=14)
# ax.set_xticklabels(columns, rotation=90, fontname='Times New Roman', fontsize=10)

# Set the y-axis label
# ax.set_yticklabels(fontname='Times New Roman', fontsize=14)
ax.set_ylabel('Temperature (K)', labelpad=10, fontname='Times New Roman', fontsize=14)

# Add legend
ax.legend(handles, labels, loc='upper right', frameon=False)

# Save Fig
plt.savefig(r'D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\Results\Box_Plot_max_1.png', bbox_inches='tight', dpi=350)

# Display the plot
plt.show()


# df_clip_prop =pd.read_csv(r"D:\UNI\M\Thesis\2\SFW_Code\#Code\#Temp_Prediction_Amsterdam\Preds\AVG\UrbClim_Prediction_BoxPlot_Properties.csv")


# Dictionary to store column properties
column_properties = {}

# Calculate statistics for each column (box plot)
for column in df.columns:
    data = df[column]
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    # outliers = [x for x in data if x < lower_fence or x > upper_fence]
    
    # Store the statistics in the dictionary
    column_properties[column] = {
        "Median": median,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence,
        # "Outliers": outliers
    }

# Create a DataFrame from the dictionary
column_properties_df = pd.DataFrame.from_dict(column_properties, orient='index')

column_properties_df.to_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\MAX\column_properties_df.csv" ,index=True)


prediction_df = column_properties_df[column_properties_df.index.str.startswith('Prediction')]
urbclim_df = column_properties_df[column_properties_df.index.str.startswith('UrbClim')]

# Get the column names to iterate over
column_names = column_properties_df.columns

# Create an empty DataFrame to store the differences
difference_df = pd.DataFrame(index=prediction_df.index)

# Calculate the difference between the Median values of each corresponding row
for column in column_names:
    difference_df[column + ' Difference'] = prediction_df[column].values - urbclim_df[column].values

# Display the resulting DataFrame
print(difference_df)





"""MIN Temp"""


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

df =pd.read_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\MIN\UrbClim_Prediction_Min_Temps_Points.csv")

df = df.drop(['OID_', 'pointid'], axis=1)

for i in range(0,12):
    df = df.rename(columns={df.columns[i]: df.columns[i][8:].replace("_"," ")})

for i in range(12,24):
    df = df.rename(columns={df.columns[i]: (df.columns[i][:8] + df.columns[i][17:]).replace("_"," ")})


df = df[['Prediction Jan 17', 'UrbClim Jan 17', 'Prediction Jan 18', 'UrbClim Jan 18', 'Prediction Jan 19', 'UrbClim Jan 19',
          'Prediction May 04', 'UrbClim May 04', 'Prediction May 05', 'UrbClim May 05', 'Prediction May 06', 'UrbClim May 06',
          'Prediction July 18', 'UrbClim July 18', 'Prediction July 19', 'UrbClim July 19', 'Prediction July 20', 'UrbClim July 20',
          'Prediction Oct 24', 'UrbClim Oct 24', 'Prediction Oct 25', 'UrbClim Oct 25', 'Prediction Oct 26', 'UrbClim Oct 26']]

df.to_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\MIN\UrbClim_Prediction_Min_Temps_Points_2.csv" ,index=False)


# Separate "Prediction" and "UrbClim" columns
prediction_cols = [col for col in df.columns if col.startswith('Prediction')]
urbclim_cols = [col for col in df.columns if col.startswith('UrbClim')]

columns = df.columns

labels_col = [col.replace('Prediction ', '').replace('UrbClim ', '') for col in columns]
labels_col_1 = list(dict.fromkeys(labels_col))

# Create box plot
fig, ax = plt.subplots(figsize=(12, 7))

boxprops_pred = dict(edgecolor='teal', facecolor='lightseagreen')
medianprops_pred = dict(color='teal')
whiskerprops_pred = dict(color='teal')
capprops_pred = dict(color='teal')

boxprops_urb = dict(edgecolor='g', facecolor='mediumseagreen')
medianprops_urb = dict(color='g')
whiskerprops_urb = dict(color='g')
capprops_urb = dict(color='g')

positions = np.arange(1, len(columns) + 1)

handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor='lightseagreen', edgecolor='teal'),
           mpatches.Rectangle((0, 0), 1, 1, facecolor='mediumseagreen', edgecolor='g')]
labels = ['Prediction', 'UrbClim']


for i, col in enumerate(columns):
    if i % 2 == 0:
        ax.boxplot(df[col], positions=[positions[i]], patch_artist=True, boxprops=boxprops_pred, whiskerprops=whiskerprops_pred,
                   capprops=capprops_pred, medianprops=medianprops_pred, widths=0.6, showfliers=False)
    else:
        ax.boxplot(df[col], positions=[positions[i]], patch_artist=True, boxprops=boxprops_urb, whiskerprops=whiskerprops_urb,
                   capprops=capprops_urb, medianprops=medianprops_urb, widths=0.6, showfliers=False)

# Set x-axis labels
ax.set_xticks(positions)
ax.set_xticklabels(labels_col, rotation=45, fontname='Times New Roman', fontsize=12)
# ax.set_xticks(positions[::2]+0.5)
# ax.set_xticklabels(labels_col_1, fontname='Times New Roman', fontsize=14)
# ax.set_xticklabels(columns, rotation=90, fontname='Times New Roman', fontsize=10)

# Set the y-axis label
# ax.set_yticklabels(fontname='Times New Roman', fontsize=14)
ax.set_ylabel('Temperature (K)', labelpad=10, fontname='Times New Roman', fontsize=14)

# Add legend
ax.legend(handles, labels, loc='upper right', frameon=False)

# Save Fig
plt.savefig(r'D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\Results\Box_Plot_min_1.png', bbox_inches='tight', dpi=350)

# Display the plot
plt.show()

# Dictionary to store column properties
column_properties = {}

# Calculate statistics for each column (box plot)
for column in df.columns:
    data = df[column]
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    # outliers = [x for x in data if x < lower_fence or x > upper_fence]
    
    # Store the statistics in the dictionary
    column_properties[column] = {
        "Median": median,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence,
        # "Outliers": outliers
    }

# Create a DataFrame from the dictionary
column_properties_df = pd.DataFrame.from_dict(column_properties, orient='index')

column_properties_df.to_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\MIN\column_properties_df.csv" ,index=True)


prediction_df = column_properties_df[column_properties_df.index.str.startswith('Prediction')]
urbclim_df = column_properties_df[column_properties_df.index.str.startswith('UrbClim')]

# Get the column names to iterate over
column_names = column_properties_df.columns

# Create an empty DataFrame to store the differences
difference_df = pd.DataFrame(index=prediction_df.index)

# Calculate the difference between the Median values of each corresponding row
for column in column_names:
    difference_df[column + ' Difference'] = prediction_df[column].values - urbclim_df[column].values

# Display the resulting DataFrame
print(difference_df)
