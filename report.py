import pandas as pd
import joblib
from sklearn.metrics import classification_report

normal_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\normal\61.44.csv", header=None, nrows=30)
v_misalignment_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\horizontal-misalignment\1.0mm\60.416.csv", header=None, nrows=30)
h_misalignment_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\vertical-misalignment\1.27mm\62.2592.csv", header=None, nrows=30)

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

X_test = data.drop(columns=['label'])  # Features
y_test = data['label']                 # Labels

model = joblib.load('Models/knn_model.pkl')
print("Test Started")

y_pred = model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred)


with open('./Reports/knn_report.txt', 'w') as f:
    f.write(report)

print("Classification report saved to 'knn_report.txt'.")