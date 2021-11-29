# comet.ml
from comet_ml import Experiment
import os
from pathlib import Path
import pickle

# data science
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

# xgboost
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb

# coding
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('../ift6758/data/milestone2')
from q3_baseline import read_dataset, plot_models

#classes for grid search and cross-validation, function for splitting data and evaluating models
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

import time

# need to install graohviz
# > conda install graphviz python-graphviz

def train(X, y, features=['Distance from Net']):
    '''
    reference: https://machinelearningmastery.com/develop-first-xgboost-model-python-scikit-learn/
    '''
    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = 'milestone_2',
        workspace='xiaoxin-zhou')    

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X[features],
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)

    # Fit model no training data
    clf = xgb.XGBClassifier()
    clf.fit(X_train, y_train)

    # save the model to disk
    filename = r'models/xgboost/q5_1.pkl'
    pickle.dump(clf, open(filename, 'wb'))

    # experiment.log_model('xgboost_5_1', filename)
    

    # Make predictions for test data
    y_pred = clf.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = metrics.accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    params = {
        'model_type': 'xgboost',
        'feature': features
    }

    metrics_dict = {
        'accuracy': accuracy
    }

    experiment.log_dataset_hash(X_train)
    experiment.log_parameters(params)
    experiment.log_metrics(metrics_dict)    

# xsticks and ysticks are unstable on jupyter environment
def q5_1_plots():
    X,y = read_dataset()
    plot_models(X, y, 'xgb')

def q5_2():
    
    # Read CSV files
    dataset = pd.read_csv('/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/all_data_categorical.csv')

    # Separate features and labels 
    X = dataset[['eventIdx', 'game_id', 'Game Seconds', 'Game Period', 'X-Coordinate', 'Y-Coordinate',
               'Shot Distance', 'Shot Angle', 'Shot Type', 'Was Net Empty', 'Last Event Type', 'Last X-Coordinate',
               'Last Y-Coordinate', 'Time from Last Event (seconds)', 'Distance from Last Event', 'Is Rebound',
               'Change in Shot Angle', 'Speed']]
    y = dataset[['Is Goal']]

    # initialize comet.ml experiment
    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = 'milestone_2',
        workspace='xiaoxin-zhou') 

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)
    
    # https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663
    params = {'n_estimators': [50, 100, 500],
              'max_depth': [3, 6, 10],
              'learning_rate': [0.01, 0.05, 0.1],# 0.2, 0.5, 0.7],
              'booster': ['gbtree', 'gblinear', 'dart'],
              
              # 'tree_method': [],  # just keep this as commented out to use its default value
              # 'gamma': [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 200],
              # 'min_child_weight': [],
              # 'max_delta_step': [],
              # 'subsample': [],
              # 'colsample_bytree': [0.3, 0.7],
              # 'colsample_bylevel': [0.3, 0.7],
              # 'colsample_bynode': [0.3, 0.7],
              # 'reg_alpha',
              # 'reg_lambda',
    }
    
    model = xgb.XGBClassifier()
    
    # When cv is not set, it defaults to 5-fold cross validation: 
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    clf = GridSearchCV(estimator=model,
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       refit=True,
                       verbose=4)
    
    clf.fit(X_train, y_train)

    print("Best parameters:", clf.best_params_)
    print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
    
    y_pred = clf.predict(X_test)
    # print('y_pred', y_pred)
    
    # Evaluate predictions
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    # save the model to disk
    filename = r'models/xgboost/q5_2.pkl'
    
    pickle.dump(clf, open(filename, 'wb'))
    
    # log model to comet.ml 
    experiment.log_model('xgboost_5_2', filename)

    # experiment.log_model('xgboost_5_1', filename)
    # Predict the probability of each test sample being of a given class
    # X_test_pred_proba = clf.predict_proba(X_test)
    # print(X_test_pred_proba)
    
    # return X_test_pred_proba, y_test, clf, X_test
    


if __name__ == '__main__':
    # X,y = read_dataset()
    # train(X, y, features=['Distance from Net','Angle from Net'])
    
    # q5_1_plots()

    # q5_2()



# What are 'model' and 'pickle'?
# - https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-using-pickle
# - https://towardsdatascience.com/why-turn-into-a-pickle-b45163007dac 