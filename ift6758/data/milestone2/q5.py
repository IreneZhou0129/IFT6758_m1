import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../ift6758/data/milestone2')
from q3_baseline import read_dataset

def train(X, y, features=['Distance from Net']):
    '''
    reference: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
    '''
    
    X,y = read_dataset()

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X[features],
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)

    # Fit model no training data
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))