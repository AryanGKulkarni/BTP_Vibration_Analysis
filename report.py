import pandas as pd
import joblib
from sklearn.metrics import classification_report
import xgboost as xgb

normal_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\normal\14.336.csv", header=None, nrows=20)
v_misalignment_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\vertical-misalignment\1.90mm\16.1792.csv", header=None, nrows=20)
h_misalignment_data = pd.read_csv(r"C:\Users\aryan\Desktop\BTP\Dataset\horizontal-misalignment\2.0mm\21.7088.csv", header=None, nrows=20)

column_names = ['tachometer_signal', 'underhang_accelerometer_axial', 'underhang_accelerometer_radial',
                'underhang_accelerometer_tangential', 'overhang_accelerometer_axial', 'overhang_accelerometer_radial',
                'overhang_accelerometer_tangential', 'microphone']

normal_data.columns = column_names
h_misalignment_data.columns = column_names
v_misalignment_data.columns = column_names
model_path = './Models/main/gnb_model.pkl'

normal_data['label'] = 0
h_misalignment_data['label'] = 1
v_misalignment_data['label'] = 2

data = pd.concat([normal_data, h_misalignment_data, v_misalignment_data], axis=0)

X_test = data.drop(columns=['label'])  # Features
y_test = data['label']                 # Labels

scaler = joblib.load('./Models/scaler.pkl') 

model = joblib.load(model_path)
X_test_scaled = scaler.transform(X_test)
print("Test Started")

# if(model_path.find("xgb")):
#     print("xgb model")
#     X_test = xgb.DMatrix(X_test)

y_pred = model.predict(X_test)

# Generate the classification report
report = classification_report(y_test, y_pred)


with open('./Reports/gnb_report.txt', 'w') as f:
    f.write(report)

print("Classification report saved to 'gnb_report.txt'.")