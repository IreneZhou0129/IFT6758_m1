# comet.ml
from comet_ml import Experiment
import os
from pathlib import Path
import pickle

# xgboost
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import xgboost as xgb

# coding
import time
import warnings
warnings.filterwarnings('ignore')

# data science
import numpy as np
import pandas as pd
from sklearn import preprocessing
#classes for grid search and cross-validation, function for splitting data and evaluating models
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, RFE, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# customized APIs
import sys
sys.path.append('../ift6758/data/milestone2')
from q6_baseline import read_all_features, plot_models

# need to install graohviz
# > conda install graphviz python-graphviz

# ##################################################
# Train XGBoost using only 'distance' and 'angle'
# ##################################################
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
    

    # Make predictions for test data
    y_pred = clf.predict(X_test)
    predictions = [round(value) for value in y_pred]

    # Evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
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

# ##################################################
# Train XGBoost using all features
# ##################################################
def q5_2(X, y, experiment):

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)
    
    # https://towardsdatascience.com/xgboost-fine-tune-and-optimize-your-model-23d996fab663
    params = {'n_estimators': [50, 100, 500],
              'max_depth': [3, 6, 10],
              'learning_rate': [0.01, 0.05, 0.1],
              'booster': ['gbtree', 'gblinear', 'dart']}
    
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
    
    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    metrics_dict = {
        'accuracy': accuracy
    }
    
    experiment.log_parameters(clf.best_params_)
    experiment.log_metrics(metrics_dict)
    
    # save the model to disk
    # filename = r'models/xgboost/q5_2.pkl'
    
    # pickle.dump(clf, open(filename, 'wb'))
    
    # log model to comet.ml 
    # experiment.log_model('xgboost_5_2', filename)

    # return None


# In above q5_2(), we did not log the experiment to model. Based on the output, we got:
# Best parameters: {
#     'booster': 'gbtree', 
#     'learning_rate': 0.05, 
#     'max_depth': 10, 
#     'n_estimators': 100}
# So, in q5_2_tuned(), we apply these parameters to a quick run and log accuracy to the experiment.
def q5_2_tuned(X, y, experiment):
    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.5,
        booster='gbtree'
    )

    params = {
        'n_estimators': [100],
        'max_depth': [10], 
        'learning_rate':  [0.05],
        'booster': ['gbtree'],
    }
    
    model.fit(X_train, y_train)

    clf = GridSearchCV(estimator=model,
                       param_grid=params,
                       scoring='neg_mean_squared_error',
                       refit=True,
                       verbose=4)
    
    clf.fit(X_train, y_train)

    # Make predictions for test data
    y_test_pred = clf.predict(X_test)
    y_test = y_test.to_numpy().flatten()
    
    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    metrics_dict = {
        'accuracy': accuracy
    }
    
    experiment.log_parameters(params)
    experiment.log_metrics(metrics_dict)

    # # save the model to disk
    model.save_model("models/q5_2_tuned.model")
    model_name = "XGBoost Model (q5_2_tuned)"
    experiment.log_model(model_name, "models/q5_2_tuned.model")
    experiment.end()    

    return None
    

# ###################################################
# Feature selection
# ref: 
# - https://scikit-learn.org/stable/modules/feature_selection.html
# ###################################################
def q5_3(X, y, experiment):

    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)
    
    model = xgb.XGBClassifier()
    
    model.fit(X_train, y_train)

    # Make predictions for test data
    y_test_pred = model.predict(X_test)
    y_test = y_test.to_numpy().flatten()
    
    # Evaluate predictions
    accuracy = accuracy_score(y_test, y_test_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # selected_features = list(X.columns)

    params = {
        'model_type': 'xgboost',
        # 'selected_features': selected_features
    }

    metrics_dict = {
        'accuracy': accuracy
    }
    
    experiment.log_parameters(params)
    experiment.log_metrics(metrics_dict)

    return model
    

# https://scikit-learn.org/stable/modules/feature_selection.html#removing-features-with-low-variance
def q5_3_var_threshold(X, y, experiment): 
    """
    Removing Features with low variance.
    """ 

    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel_reduced = sel.fit_transform(X)
    
    model = q5_3(sel_reduced, y, experiment)
    
    experiment.log_dataset_hash(sel_reduced)
    
    # https://github.com/comet-ml/comet-examples/blob/master/model_registry/xgboost_seldon_aws/xgboost_seldon_aws.ipynb
    model.save_model("models/q5_3_var_threshold.model")
    model_name = "XGBoost Model (var threshold)"
    experiment.log_model(model_name, "models/q5_3_var_threshold.model")
    experiment.end()


def q5_3_selectKbest(X, y, experiment):  
    """
    Univariate Feature Selection.
    ref:
    - https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
    - https://machinelearningmastery.com/feature-selection-machine-learning-python/ 
    """
    
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in X:
        if X[col].dtypes != np.object:
            # https://www.kite.com/python/answers/how-to-scale-pandas-dataframe-columns-with-the-scikit-learn-minmaxscaler-in-python
            X[[col]] = min_max_scaler.fit_transform(X[[col]])

    X_new = SelectKBest(chi2, k=6).fit_transform(X, y)
    
    model = q5_3(X_new, y, experiment)
    
    experiment.log_dataset_hash(X_new)
    
    # https://github.com/comet-ml/comet-examples/blob/master/model_registry/xgboost_seldon_aws/xgboost_seldon_aws.ipynb
    # model.save_model("models/q5_3_selectKBest.model")
    # model_name = "XGBoost Model (q5_3_selectKbest)"
    # experiment.log_model(model_name, "models/q5_3_selectKBest.model")

    model.save_model("models/q5_3_selectKBest.pkl")
    model_name = "XGBoost Model (q5_3_selectKbest)"
    experiment.log_model(model_name, "models/q5_3_selectKBest.pkl")

    experiment.end()


# https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel
def q5_3_selectFromModel(X, y, experiment):
    """
    L1-based Feature Selection.
    """
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    
    model = q5_3(X_new, y, experiment)
    
    experiment.log_dataset_hash(X_new)
    
    # https://github.com/comet-ml/comet-examples/blob/master/model_registry/xgboost_seldon_aws/xgboost_seldon_aws.ipynb
    model.save_model("models/q5_3_selectFromModel.model")
    model_name = "XGBoost Model (selectFromModel)"
    experiment.log_model(model_name, "models/q5_3_selectFromModel.model")
    experiment.end()


def q5_3_extraTree(X, y, experiment): 

    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(X, y)

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    
    model = q5_3(X_new, y, experiment)
    
    experiment.log_dataset_hash(X_new)
    
    # https://github.com/comet-ml/comet-examples/blob/master/model_registry/xgboost_seldon_aws/xgboost_seldon_aws.ipynb
    model.save_model("models/q5_3_extraTree.model")
    model_name = "XGBoost Model (extraTree)"
    experiment.log_model(model_name, "models/q5_3_extraTree.model")
    experiment.end()


# ###################################################
# 2.0 Feature selection
# Technique 1: q5_3_var_threshold()
# Technique 2: q5_3_selectKbest()
# Technique 3: q5_3_selectFromModel()
# Technique 4: q5_3_rfe_logreg()
# Technique 5: q5_3_extraTree()
# ###################################################
def q5_3_rfe_logreg(X, y, experiment):
    """
    Recursive feature elimination.
    """
    model = LogisticRegression(solver='lbfgs')

    rfe = RFE(estimator=model, n_features_to_select=14)
    fit = rfe.fit(X, y)
    X_new = fit.transform(X)

    model = q5_3(X_new, y, experiment)

    experiment.log_dataset_hash(X_new)
    
    # https://github.com/comet-ml/comet-examples/blob/master/model_registry/xgboost_seldon_aws/xgboost_seldon_aws.ipynb
    model.save_model("models/q5_3_rfe_logreg.model")
    model_name = "XGBoost Model (RFE)"
    experiment.log_model(model_name, "models/q5_3_rfe_logreg.model")
    experiment.end()


def main():
    X,y = read_all_features()

    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = 'milestone_2',
        workspace='xiaoxin-zhou') 

    # =================
    # Question 1
    # =================
    # train(X, y)
    # q5_1_plots()

    # =================
    # Question 2
    # =================
    # q5_2(X, y, experiment)
    # q5_2_tuned(X, y, experiment)

    # =================
    # Question 3
    # =================
    # q5_3_var_threshold(X, y, experiment)    
    q5_3_selectKbest(X, y, experiment)
    # q5_3_selectFromModel(X, y, experiment)
    # q5_3_extraTree(X, y, experiment)
    # q5_3_rfe_logreg(X, y, experiment)

if __name__ == '__main__':
    main()



# What are 'model' and 'pickle'?
# - https://practicaldatascience.co.uk/machine-learning/how-to-save-and-load-machine-learning-models-using-pickle
# - https://towardsdatascience.com/why-turn-into-a-pickle-b45163007dac 