# comet_ml
from comet_ml import API
from pathlib import Path
import pickle

# data science
import pandas as pd 
import numpy as np 

# customized APIs
from q2 import q2_filter_data
from q4 import q4_categorical
from q6_baseline import read_all_features

# coding
import time
import warnings
warnings.filterwarnings('ignore')

# apply advanced feature engineering on test data 
def create_csv(fe_type, file_path, dest_path, years):

    # part 2: Feature Engineering I
    if fe_type == 'simple':
        q2_filter_data(file_path, dest_path, years)

    else:
        # part 4: Feature Engineering II
        q4_categorical(file_path, dest_path, years)

    return None


def main(X, y):
    """
    Four figures: ROC, Goal rate, Cumulative proportion, Calibration curve
    5 curves on each figure: 
        (1). logreg on distance only
        (2). logreg on angle only
        (3). logreg on both distance and angle
        (4). best XGBoost in part 5: "Technique 2: Univariate Feature Selection"
        (5). best model in part 6: "Approach 2:
            Decision Tree Classifier with Randomized search on hyper parameters and Regularization"
    """   
    # for curve 1, 2, and 3
    q2_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_simple.csv'

    # for curve 4 and 5
    q4_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_categorical.csv'     

    five_curves = {
        '1':{
            'pickle_name': 'models\log_reg\log_reg_dist.pkl',
            'all_feature': False,
            'feature': ['Distance from Net'],
            'model_type': 'logreg',
            'data_path': q2_path,
            'color': 'r'
        },
        '2':{
            'pickle_name': 'models\log_reg\log_reg_angle.pkl',
            'all_feature': False,
            'feature': ['Angle from Net'],
            'model_type': 'logreg',
            'data_path': q2_path,
            'color': 'g'    
        },
        '3':{
            'pickle_name': 'models\log_reg\log_reg_both.pkl',
            'all_feature': False,
            'feature': ['Distance from Net', 'Angle from Net'],
            'model_type': 'logreg',
            'data_path': q2_path,
            'color': 'b',              
        },
        # '4':{
        #     'pickle_name': 'q5_3_selectKBest.pkl',
        #     'all_feature': True,
        #     'model_type': 'xgb_tech_2',
        #     'data_path': q4_path,
        # 'color': 'r'
        # },
        '5':{
            'pickle_name': 'decision_tree/approach_2.pkl',
            'all_feature': True,
            'model_type': 'approach_2',
            'data_path': q4_path,
            'color': 'plum'               
        }
    }

    # load model
    model_full_path = '/Users/xiaoxinzhou/Documents/2021-Fall/UdeM/IFT6758/IFT6758_m1/models'

    
    # read X, y
    X, y = read_all_features(five_curves[k]['data_path'])

    score = model.score(X, y)
    print(pickle_name, score)

    #--------------------------------------
    # ROC
    #--------------------------------------
    plot_roc(X, y, five_curves)
    plt.show() 

    #--------------------------------------
    # Goal rate
    #--------------------------------------  
    plot_goal_rate(X, y, five_curves)
    plt.show()
    
    #--------------------------------------
    # Cumulative proportion of goals
    #--------------------------------------  
    plot_cumulative_rate(X, y, five_curves)
    plt.show()
    
    #--------------------------------------
    # Calibration
    #-------------------------------------- 
    plot_calibration(X, y, five_curves)
    plt.show()    
    


if __name__ == '__main__':
    # ==============================
    # create test dataset csv
    # ==============================
    
    file_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_JSON_data/regular_season'

    # for curve 1, 2, and 3
    q2_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_simple.csv'
    # for curve 4 and 5
    dest_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_categorical.csv'
    years = [2019]

    # create_csv('simple', file_path, dest_path, years)

    X, y = read_all_features(dest_path)
    main(X,y)


    
    # 1. q3 baseline distance only
    #    score: 0.902948905109489
    # pickle_name = '/models\log_reg\log_reg_dist.pkl'
    # main(pickle_name, X[['Distance from Net']], y)
    
    # 2. q3 baseline angle only
    #    score: 0.902948905109489
    # pickle_name = '/models\log_reg\log_reg_angle.pkl'
    # main(pickle_name, X[['Angle from Net']], y)

    # 3. q3 baseline distance and angle
    #    score: 0.902948905109489
    # pickle_name = '/models\log_reg\log_reg_both.pkl'
    # main(pickle_name, X[['Distance from Net', 'Angle from Net']], y)
    

    