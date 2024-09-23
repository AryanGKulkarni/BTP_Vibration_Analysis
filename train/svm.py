import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

normal_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\normal\12.288.csv", header=None, nrows=200)
v_misalignment_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\vertical-misalignment\1.90mm\12.0832.csv", header=None, nrows=200)
h_misalignment_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\horizontal-misalignment\2.0mm\12.288.csv", header=None, nrows=200)

column_names = ['tachometer_signal', 'underhang_accelerometer_axial', 'underhang_accelerometer_radial',
                'underhang_accelerometer_tangential', 'overhang_accelerometer_axial', 'overhang_accelerometer_radial',
                'overhang_accelerometer_tangential', 'microphone']

normal_data.columns = column_names
h_misalignment_data.columns = column_names
v_misalignment_data.columns = column_names

normal_data['label'] = 0
h_misalignment_data['label'] = 1
v_misalignment_data['label'] = 2

data = pd.concat([normal_data, h_misalignment_data, v_misalignment_data], axis=0)

from sklearn.model_selection import train_test_split

X_train = data.drop(columns=['label'])  # Features
y_train = data['label']                 # Labels

from sklearn import svm

print("Training Started")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, './Models/scaler.pkl')

svmmodel = svm.SVC()
svmmodel.fit(X_train_scaled, y_train)

print("Training Done")

import joblib

# Save the model to a file
joblib.dump(svmmodel, './Models/svm_model.pkl')
print("Model Saved")