import pandas as pd
import os
import random
import data_loader

X_train, y_train = data_loader.load_data()
print("Training Started")

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    min_samples_split=2, random_state=0)
clf.fit(X_train, y_train)

print("Training Done")

import joblib

# Save the model to a file
joblib.dump(clf, './Models/rf_model.pkl')
print("Model Saved")