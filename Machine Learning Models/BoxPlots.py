"""
@author: FatemehChajaei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Box Plot Function
def create_box_plot(df, save_path):
    # Change Columns Name
    
    # df = df.drop(['OID_'], axis=1)

    # for i in range(0,12):
    #     df = df.rename(columns={df.columns[i]: df.columns[i][8:].replace("_"," ")})

    # for i in range(12,24):
    #     df = df.rename(columns={df.columns[i]: (df.columns[i][:8] + df.columns[i][17:]).replace("_"," ")})

    # df = df[['Prediction Jan 17', 'UrbClim Jan 17', 'Prediction Jan 18', 'UrbClim Jan 18', 'Prediction Jan 19', 'UrbClim Jan 19',
    #           'Prediction May 04', 'UrbClim May 04', 'Prediction May 05', 'UrbClim May 05', 'Prediction May 06', 'UrbClim May 06',
    #           'Prediction July 18', 'UrbClim July 18', 'Prediction July 19', 'UrbClim July 19', 'Prediction July 20', 'UrbClim July 20',
    #           'Prediction Oct 24', 'UrbClim Oct 24', 'Prediction Oct 25', 'UrbClim Oct 25', 'Prediction Oct 26', 'UrbClim Oct 26']]

    labels_col = [col.replace('Prediction ', '').replace('UrbClim ', '') for col in df.columns]

    fig, ax = plt.subplots(figsize=(12, 7))

    boxprops_pred = dict(edgecolor='teal', facecolor='lightseagreen')
    medianprops_pred = dict(color='teal')
    whiskerprops_pred = dict(color='teal')
    capprops_pred = dict(color='teal')

    boxprops_urb = dict(edgecolor='g', facecolor='mediumseagreen')
    medianprops_urb = dict(color='g')
    whiskerprops_urb = dict(color='g')
    capprops_urb = dict(color='g')

    positions = np.arange(1, len(df.columns) + 1)

    handles = [mpatches.Rectangle((0, 0), 1, 1, facecolor='lightseagreen', edgecolor='teal'),
               mpatches.Rectangle((0, 0), 1, 1, facecolor='mediumseagreen', edgecolor='g')]
    labels = ['Prediction', 'UrbClim']

    for i, col in enumerate(df.columns):
        if i % 2 == 0:
            ax.boxplot(df[col], positions=[positions[i]], patch_artist=True,
                       boxprops=boxprops_pred, whiskerprops=whiskerprops_pred,
                       capprops=capprops_pred, medianprops=medianprops_pred,
                       widths=0.6, showfliers=False)
        else:
            ax.boxplot(df[col], positions=[positions[i]], patch_artist=True,
                       boxprops=boxprops_urb, whiskerprops=whiskerprops_urb,
                       capprops=capprops_urb, medianprops=medianprops_urb,
                       widths=0.6, showfliers=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_col, rotation=45, fontname='Times New Roman', fontsize=12)
    ax.set_ylabel('Temperature (K)', labelpad=10, fontname='Times New Roman', fontsize=14)
    ax.legend(handles, labels, loc='upper right', frameon=False)

    plt.savefig(save_path, bbox_inches='tight', dpi=350)
    plt.show()


# Calculate Statistics Function
def calculate_statistics(df):
    # Dictionary to store column properties
    column_properties = {}

    for column in df.columns:
        data = df[column]
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        # outliers = [x for x in data if x < lower_fence or x > upper_fence]

        column_properties[column] = {
            "Median": median,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower Fence": lower_fence,
            "Upper Fence": upper_fence,
            # "Outliers": outliers
        }

    return pd.DataFrame.from_dict(column_properties, orient='index')


# Calculate Differences Function
def calculate_differences(prediction_df, urbclim_df):
    difference_df = pd.DataFrame(index=prediction_df.index)

    for column in prediction_df.columns:
        difference_df[column + ' Difference'] = prediction_df[column].values - urbclim_df[column].values

    return difference_df

# Load Data
avg_file_path = r"D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\UrbClim_Prediction_Temps_Points_2.csv"
max_file_path = r"D:\Code\Temp_Prediction_Amsterdam\Preds\MAX\UrbClim_Prediction_Temps_Points.csv"
min_file_path = r"D:\Code\Temp_Prediction_Amsterdam\Preds\MIN\UrbClim_Prediction_Min_Temps_Points.csv"

# Load and Generate Box Plot for Average
df_avg = pd.read_csv(avg_file_path)
create_box_plot(df_avg, r'D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\Results\Box_Plot_1.png')

# Load and Generate Box Plot for Maximum
df_max = pd.read_csv(max_file_path)
create_box_plot(df_max, r'D:\Code\Temp_Prediction_Amsterdam\Preds\MAX\Results\Box_Plot_1.png')

# Load and Generate Box Plot for Minimum
df_min = pd.read_csv(min_file_path)
create_box_plot(df_min, r'D:\Code\Temp_Prediction_Amsterdam\Preds\MIN\Results\Box_Plot_1.png')

# Generate Statistics for Average
statistics_df_avg = calculate_statistics(df_avg)
statistics_df_avg.to_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\AVG\column_properties_df.csv", index=True)

# Generate Statistics for Maximum
statistics_df_max = calculate_statistics(df_max)
statistics_df_max.to_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\MAX\column_properties_df.csv", index=True)

# Generate Statistics for Minimum
statistics_df_min = calculate_statistics(df_min)
statistics_df_min.to_csv(r"D:\Code\Temp_Prediction_Amsterdam\Preds\MIN\column_properties_df.csv", index=True)

# Generate Differences for Average
prediction_df_avg = statistics_df_avg[statistics_df_avg.index.str.startswith('Prediction')]
urbclim_df_avg = statistics_df_avg[statistics_df_avg.index.str.startswith('UrbClim')]
difference_df_avg = calculate_differences(prediction_df_avg, urbclim_df_avg)
print(difference_df_avg)

# Generate Differences for Maximum
prediction_df_max = statistics_df_max[statistics_df_max.index.str.startswith('Prediction')]
urbclim_df_max = statistics_df_max[statistics_df_max.index.str.startswith('UrbClim')]
difference_df_max = calculate_differences(prediction_df_max, urbclim_df_max)
print(difference_df_max)

# Generate Differences for Minimum
prediction_df_min = statistics_df_min[statistics_df_min.index.str.startswith('Prediction')]
urbclim_df_min = statistics_df_min[statistics_df_min.index.str.startswith('UrbClim')]
difference_df_min = calculate_differences(prediction_df_min, urbclim_df_min)
print(difference_df_min)
