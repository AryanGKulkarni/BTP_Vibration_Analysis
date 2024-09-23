import pandas as pd

normal_data = pd.read_csv('./Datasets/normal.csv', header=None, nrows=2000)
v_misalignment_data = pd.read_csv('./Datasets/vertical-misalignment.csv', header=None, nrows=2000)
h_misalignment_data = pd.read_csv('./Datasets/horizontal-misalingment.csv', header=None, nrows=2000)

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

X = data.drop(columns=['label'])  # Features
y = data['label']                 # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn import svm

print("Training Started")

svmmodel = svm.SVC()
svmmodel.fit(X_train, y_train)

print("Training Done")

import joblib

# Save the model to a file
joblib.dump(svmmodel, './Models/svm_model.pkl')
print("Model Saved")