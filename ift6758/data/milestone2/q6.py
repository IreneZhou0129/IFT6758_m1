# comet.ml
import os
import pickle
from comet_ml import Experiment
from pathlib import Path

# data science
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc

# ift6758/data/milestone2/q6_baseline
import sys
sys.path.append('../ift6758/data/milestone2')
from q6_baseline import read_all_features, plot_models

# create a decorator
def log_experiment(func):
    def wrapper(*args, **kwargs):

        experiment = Experiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name='milestone_2',
            workspace='xiaoxin-zhou')

        X_train = args[0]
        model_type = args[4]
        file_name = args[5]

        clf, params, metrics_dict = func(*args, **kwargs)

        # create file path if it doesn't exist
        dirname = os.path.dirname(file_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # save the model to disk
        pickle.dump(clf, open(file_name, 'wb'))

        experiment.log_model(model_type, file_name)

        experiment.log_dataset_hash(X_train)
        experiment.log_parameters(params)
        experiment.log_metrics(metrics_dict)

    return wrapper


##############################################################################
# Approch 1: Decision Tree Classifier
##############################################################################
@log_experiment
def approach_1(X_train, X_test, y_train, y_test, model_type, file_name):
    """
    :param model_type: 'decision_tree'
    :param file_name:: 'models/decision_tree/approach_1.pkl'
    """

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, thr = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    metrics_dict = {
        'model_type': model_type,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    params = {
        'params': 'Default Values'
    }

    return clf, params, metrics_dict


###################################################################################################
# Approach 2: Decision Tree Classifier with
# Randomized search on hyper parameters and Regularization
###################################################################################################
@log_experiment
def approach_2(X_train, X_test, y_train, y_test, model_type, file_name):
    """
    https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/

    :param model_type: 'decision_tree'
    :param file_name:: 'models/decision_tree/approach_2.pkl'
    """

    dtc = DecisionTreeClassifier()

    space = dict()
    space["splitter"] = ["best", "random"]
    space["max_depth"] = list(range(2, 50))
    space["min_samples_split"] = np.linspace(0.1, 1.0, 10, endpoint=True)
    space["min_samples_leaf"] = np.linspace(0.1, 0.5, 5, endpoint=True)
    space["max_features"] = list(range(1, X_train.shape[1]))
    space["max_leaf_nodes"] = list(range(2, 10))

    clf = RandomizedSearchCV(dtc, space, random_state=50, verbose=3)

    search = clf.fit(X_train, y_train)
    y_pred = search.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, thr = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    metrics_dict = {
        'model_type': model_type,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    params = clf.best_params_

    return clf, params, metrics_dict


####################################################################################
# Approach 3: Decision Tree Classifier with PCA Feature Selection
####################################################################################
@log_experiment
def approach_3(X_train, X_test, y_train, y_test, model_type, file_name):
    """
    :param model_type: 'decision_tree'
    :param file_name:: 'models/decision_tree/approach_3.pkl'
    """
    pca = PCA(n_components=3)
    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.fit_transform(X_test)

    clf = DecisionTreeClassifier()
    clf.fit(X_train_transformed, y_train)

    y_pred = clf.predict(X_test_transformed)

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, thr = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    metrics_dict = {
        'model_type': model_type,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    params = {
        'params': 'Default Values'
    }

    return clf, params, metrics_dict


####################################################################################
# Approach 4: Decision Tree Classifier combining approaches 2 and 3
####################################################################################
@log_experiment
def approach_4(X_train, X_test, y_train, y_test, model_type, file_name):
    """
    https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/

    :param model_type: 'decision_tree'
    :param file_name:: 'models/decision_tree/approach_3.pkl'
    """

    pca = PCA(n_components=3)
    X_train_transformed = pca.fit_transform(X_train)
    X_test_transformed = pca.fit_transform(X_test)

    dtc = DecisionTreeClassifier()

    space = dict()
    space['splitter'] = ['best', 'random']
    space['max_depth'] = list(range(2, 50))
    space['min_samples_split'] = np.linspace(0.1, 1.0, 10, endpoint=True)
    space['min_samples_leaf'] = np.linspace(0.1, 0.5, 5, endpoint=True)
    space['max_features'] = list(range(1, X_train.shape[1]))
    space['max_leaf_nodes'] = list(range(2, 10))

    clf = RandomizedSearchCV(dtc, space, random_state=50, verbose=3)

    search = clf.fit(X_train_transformed, y_train)

    y_pred = search.predict(X_test_transformed)

    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, thr = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    metrics_dict = {
        'model_type': model_type,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    params = clf.best_params_

    return clf, params, metrics_dict


def main():
    X, y = read_all_features()

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)
    
    approach_1(X_train, X_test,
               y_train, y_test,
               'decision_tree',
               'models/decision_tree/approach_1.pkl')

    
    approach_2(X_train, X_test,
               y_train, y_test,
               'decision_tree',
               'models/decision_tree/approach_2.pkl')

    
    approach_3(X_train, X_test,
               y_train, y_test,
               'decision_tree',
               'models/decision_tree/approach_3.pkl')

    
    approach_4(X_train, X_test,
               y_train, y_test,
               'decision_tree',
               'models/decision_tree/approach_4.pkl')    

if __name__ == '__main__':

    main()
