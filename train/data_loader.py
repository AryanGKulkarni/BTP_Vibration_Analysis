import pandas as pd
import os
import random

# Define column names
column_names = ['tachometer_signal', 'underhang_accelerometer_axial', 'underhang_accelerometer_radial',
                'underhang_accelerometer_tangential', 'overhang_accelerometer_axial', 'overhang_accelerometer_radial',
                'overhang_accelerometer_tangential', 'microphone']

normal_folder = r"C:\Users\aryan\Desktop\BTP\Dataset\normal"
h_misalignment_folder = r"C:\Users\aryan\Desktop\BTP\Dataset\horizontal-misalignment\2.0mm"
v_misalignment_folder = r"C:\Users\aryan\Desktop\BTP\Dataset\vertical-misalignment\1.90mm"

def load_random_files(folder_path, label, n=45):
    files = os.listdir(folder_path)
    selected_files = random.sample(files, n)  # Randomly select n files
    data_list = []
    for file in selected_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=None, nrows=30)  # Read first 30 rows
        df.columns = column_names  # Assign only the 8 feature columns
        df['label'] = label  # Add the label column
        data_list.append(df)
    return pd.concat(data_list, axis=0)

def load_data():
    # Load data from each folder
    normal_data = load_random_files(normal_folder, label=0)
    h_misalignment_data = load_random_files(h_misalignment_folder, label=1)
    v_misalignment_data = load_random_files(v_misalignment_folder, label=2)

    # Concatenate data
    data = pd.concat([normal_data, h_misalignment_data, v_misalignment_data], axis=0)

    # Separate features and labels
    X = data.drop(columns=['label'])  # Features
    y = data['label']                 # Labels

    return X, y
