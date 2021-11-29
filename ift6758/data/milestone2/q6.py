# comet.ml
from comet_ml import Experiment
import os
from pathlib import Path
import pickle

# data science
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ift6758/data/milestone2/q3_baseline
import sys
sys.path.append('../ift6758/data/milestone2')
from q3_baseline import plot_models
    
def read_all_features():
    dataset = pd.read_csv('/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/all_data_categorical.csv')
    X = dataset.iloc[: , :-1]
    y = dataset[['Is Goal']]    

    return X, y

##############################################################################
# Approch 1: Decision Tree Classifier
##############################################################################

def tree_train(X, y):
    
    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = 'milestone_2',
        workspace='xiaoxin-zhou')

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)    
    
    # hyperparameters
    max_leaf_nodes = 3
    max_depth = 30

    clf = DecisionTreeClassifier(
                        max_leaf_nodes=max_leaf_nodes, 
                        max_depth=max_depth)
    clf.fit(X_train, y_train)

    # save the model to disk
    filename = 'models/decision_tree/test_3.pkl'
    pickle.dump(clf, open(filename, 'wb'))  

    experiment.log_model('decision_tree', filename)  
    
    y_pred = clf.predict(X_test)
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # print("Accuracy: %.2f%%" % (accuracy * 100.0))  
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)    
    # print(f"roc_auc: {roc_auc}")

    params = {
        'model_type': 'decision_tree',
        'max_leaf_nodes': max_leaf_nodes,
        'max_depth': max_depth
    }

    metrics_dict = {
        'accuracy': accuracy,
        'roc_auc': roc_auc
    }

    experiment.log_dataset_hash(X_train)
    experiment.log_parameters(params)
    experiment.log_metrics(metrics_dict)      
    

if __name__ == '__main__':
    X,y = read_all_features()
    tree_train(X, y)