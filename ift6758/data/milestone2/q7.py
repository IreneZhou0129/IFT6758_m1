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

# apply advanced feature engineering on test data 
def create_csv(fe_type, file_path, dest_path, years):

    # part 2: Feature Engineering I
    if fe_type == 'simple':
        q2_filter_data(file_path, dest_path, years)

    else:
        # part 4: Feature Engineering II
        q4_categorical(file_path, dest_path, years)

    return None

def main(pickle_name, X, y):

    # load model
    model_full_path = '/Users/xiaoxinzhou/Documents/2021-Fall/UdeM/IFT6758/IFT6758_m1/models'

    model_path = f'{model_full_path}/{pickle_name}'

    model = pickle.load(open(model_path, 'rb'))

    score = model.score(X, y)
    print(score)


if __name__ == '__main__':
    # ==============================
    # create test dataset csv
    # ==============================
    file_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_JSON_data/regular_season'
    dest_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_simple.csv'
    years = [2019]

    # create_csv('simple', file_path, dest_path, years)

    # ==============================
    # split test dataset into features and target
    # ==============================
    X, y = read_all_features(dest_path)


    
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
    

    