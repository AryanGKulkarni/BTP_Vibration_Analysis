import pandas as pd
import os
import random

normal_folder = r"C:\Users\aryan\Desktop\BTP\Dataset\normal"
h_misalignment_folder = r"C:\Users\aryan\Desktop\BTP\Dataset\horizontal-misalignment\2.0mm"
v_misalignment_folder = r"C:\Users\aryan\Desktop\BTP\Dataset\vertical-misalignment\1.90mm"


column_names = ['tachometer_signal', 'underhang_accelerometer_axial', 'underhang_accelerometer_radial',
                'underhang_accelerometer_tangential', 'overhang_accelerometer_axial', 'overhang_accelerometer_radial',
                'overhang_accelerometer_tangential', 'microphone']

def load_random_files(folder_path, label, n=45):
    files = os.listdir(folder_path)
    selected_files = random.sample(files, n)  # Randomly select 25 files
    data_list = []
    for file in selected_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=None, nrows=30)  # Read first 30 rows
        df.columns = column_names  # Assign only the 8 feature columns
        df['label'] = label  # Add the label column separately
        data_list.append(df)
    return pd.concat(data_list, axis=0)

normal_data = load_random_files(normal_folder, label=0)
h_misalignment_data = load_random_files(h_misalignment_folder, label=1)
v_misalignment_data = load_random_files(v_misalignment_folder, label=2)


normal_data['label'] = 0
h_misalignment_data['label'] = 1
v_misalignment_data['label'] = 2

data = pd.concat([normal_data, h_misalignment_data, v_misalignment_data], axis=0)

from sklearn.model_selection import train_test_split

X_train = data.drop(columns=['label'])  # Features
y_train = data['label']                 # Labels


print("Training Started")

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print("Training Done")

import joblib

# Save the model to a file
joblib.dump(gnb, './Models/gnb_model.pkl')
print("Model Saved")