import pandas as pd
import os
import random
import data_loader
from sklearn.neural_network import MLPClassifier
import joblib

X_train, y_train = data_loader.load_data()

print("Training Started")

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train, y_train)

print("Training Done")

# Save the model to a file
joblib.dump(mlp, './Models/mlp_model.pkl')
print("Model Saved")