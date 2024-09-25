import pandas as pd
import os
import random
import data_loader
from sklearn.neural_network import MLPClassifier
import joblib

X_train, y_train = data_loader.load_data()

print("Training Started")

import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
param = {'max_depth': 6, 'eta': 0.1, 'objective': 'multi:softmax', 'num_class': 3}
num_round = 10
bst = xgb.train(param, dtrain, num_round)

print("Training Done")

# Save the model to a file
joblib.dump(bst, './Models/test/xgb_model.pkl')
print("Model Saved")