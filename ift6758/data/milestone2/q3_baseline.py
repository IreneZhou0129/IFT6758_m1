from comet_ml import Experiment
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay

def read_dataset():

    # Read CSV files
    dataset = pd.read_csv('/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/all_data.csv')

    # Separate features and labels 
    X = dataset[['eventIdx', 'game_id', 'Distance from Net',
                'Angle from Net', 'Is Net Empty']]
    y = dataset[['Is Goal']]

    return X,y


def train(model_name, model_path, features=['Distance from Net']):

    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = 'milestone_2',
        workspace='xiaoxin-zhou')

    Path('models/').mkdir(exist_ok=True)
    experiment.log_model(model_name, model_path)

    X, y = read_dataset()

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X[features],
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)

    # Logistic regression model fitting
    clf = LogisticRegression()
    y_train = y_train.values.ravel()
    clf.fit(X_train, y_train)

    # Predict on validation set
    y_pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)

    params = {
        'model_type': 'log_regression',
        'feature': model_name
    }

    metrics_dict = {
        'accuracy': accuracy
    }

    experiment.log_dataset_hash(X_train)
    experiment.log_parameters(params)
    experiment.log_metrics(metrics_dict)

train('distance_and_angle', 
        'models/distance_and_angle.h5', 
        features=['Distance from Net', 'Angle from Net'])    