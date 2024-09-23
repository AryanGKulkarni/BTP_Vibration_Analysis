import pandas as pd
import os
import random
import data_loader

X_train, y_train = data_loader.load_data()
print("Training Started")

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

print("Training Done")

import joblib

# Save the model to a file
joblib.dump(clf, './Models/dt_model.pkl')
print("Model Saved")