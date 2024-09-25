import pandas as pd
import os
import random
import data_loader
import joblib

X_train, y_train = data_loader.load_data()

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

print("Training Started")

from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

joblib.dump(scaler, './Models/scaler.pkl')


knnmodel = KNeighborsClassifier(n_neighbors=5)
knnmodel.fit(X_train_scaled, y_train)

print("Training Done")


# Save the model to a file
joblib.dump(knnmodel, './Models/test/knn_model.pkl')
print("Model Saved")