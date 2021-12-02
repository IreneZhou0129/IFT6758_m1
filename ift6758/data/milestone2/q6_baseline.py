from comet_ml import Experiment
import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay
import xgboost as xgb
from matplotlib.gridspec import GridSpec

def read_all_features(path='/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/all_data_categorical.csv'):
    dataset = pd.read_csv(path)
    
    # @Jason: 
    # dataset = pd.read_csv('/Users/sunjiaao/Courses/IFT6758/m2_CSV_data/all_data_q4_categorical.csv')
    
    X = dataset.iloc[: , :-1]
    y = dataset[['Is Goal']]    

    return X, y

def get_prob(X, y, model_type):
    '''
     Calculate the probability.
     In X_test_pred_proba, the first column is label 0, the second one is label 1.

     :param model_type: Logistic regression or XGBoost
    '''
    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=50)
    
    if model_type == 'decision_tree':
        clf = DecisionTreeClassifier(
                max_leaf_nodes=3, 
                max_depth=30)  

    clf.fit(X_train, y_train)

    # Predict the probability
    X_test_pred_proba = clf.predict_proba(X_test)
    
    return X_test_pred_proba

def get_percentile(X, y, probs):
    '''
    Return a df that has four columns:
        * 'index': index in y_test.
        * 'Goal prob': Goal probability.
        * 'Is Goal': 0 means not goal; 1 means goal.
        * 'Percentile': Calculated percentile, range from 0.00 to 99.99.
    The df is sorted by 'Percentile'.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=50)

    # Create a df for shot probabilities
    probs_df = pd.DataFrame(probs)
    probs_df = probs_df.rename(columns={0: "Not Goal prob", 1: "Goal prob"})

    # Concatenate 'Goal Probability' and 'Is Goal' into one df. Each column has values: 0 and 1. 
    df = pd.concat([probs_df["Goal prob"].reset_index(drop=True),\
                    y_test["Is Goal"].reset_index(drop=True)],axis=1)

    # Add 'Percentile' column
    percentile_series=df['Goal prob'].rank().apply(lambda x: 100.0*(x-1)/df.shape[0])
    df['Percentile'] = percentile_series

    # Sort the df by the value of percentile. Min is 0.0000, max is 99.9967
    df = df.sort_values(by=['Percentile'])
    df = df.reset_index()
    
    return df

def get_rate(df, function_type='goal_rate'):
    '''
    Divide the sorted df into 10 bins, each two adjacent bins have a difference of 10 on percentile.
    As we noticed, the percentile ranges from 0 to 99.99, we set bin range from 0 to 100, 
    so, there are 10 bins: [0:10], [10:20], ..., [90:100].
    Given function type, calculates the corresponding rate. 
    
    :params df: A sorted percentile df, calculated by get_percentile().
    :params function_type: Function type. Default value 'goal_rate'. 
                           Another option is 'cumulative_rate'.
    
    :return goal_rate_df: A df has shape (10, 2). First column is rate, second column is bin index. 
    '''
    
    # In goal_rate function, the list should have a length of 10, 
    # each item is the goal rate of that bin. 
    # For example, if there are 5000 records have percentiles between 0 and 10, 
    # and there are 250 goals, the goal rate is 250/5000=0.05.
    # So, rate_list[0] is 0.05
    rate_list = []
    
    # Find total number of goals
    total_goals = df['Is Goal'].value_counts()[1]
    
    cumulative_counts = 0

    i = 0
    i_list = []
    
    # 91 is the lower bound of last bin
    while i<92:
        i_list.append(i)

        # current bin size
        lower_bound = i
        upper_bound = (i+10)

        # find all records that have percentiles fall in this range
        rows = df[(df['Percentile']>=lower_bound) &
                  (df['Percentile']<upper_bound)]
        
        # if no rows exist
        if rows.empty:
            rate = 0
        else:
            # count the number of goals
            goals = rows['Is Goal'].value_counts()[1]

            shots = rows.shape[0]

            if function_type == 'goal_rate':
                rate = goals/shots
                
            elif function_type == 'cumulative_rate':
                cumulative_counts += goals
                rate = (cumulative_counts)/total_goals

        rate_list.append(rate)

        i+=10

    rate_list = [i*100 for i in rate_list]    
    
    # Combine goal rate list and percentile list into one df
    rate_df = pd.DataFrame(list(zip(rate_list, i_list)),\
                                columns=['Rate', 'Percentile'])
    
    return rate_df    


# ==========================================================================================
# Question 3:
# ==========================================================================================
def plot_roc(X, y, model_type):

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])    
        

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=50)
    probs = get_prob(X, y, model_type)
    is_goal = probs[:,1]  
    # curve_label = f[0] if len(f) == 1 else 'Distance and Angle from Net'              

    
    fpr, tpr, threshold = metrics.roc_curve(y_test, is_goal)
    
    roc_auc = metrics.auc(fpr, tpr)
    
    plt.plot(
        fpr, 
        tpr)
        # label = f'{curve_label} '+'AUC = %0.2f' % roc_auc)      
    
    plt.axis([0, 1, 0, 1])    
        
    plt.title('ROC Curves', fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)    
    plt.grid(True)


def plot_goal_rate(X, y, model_type):
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])    

    probs = get_prob(X, y, model_type)
    is_goal = probs[:,1]   
    perc_df = get_percentile(X, y, probs)
    # curve_label = f[0] if len(f) == 1 else 'Distance and Angle from Net'              

    goal_rate_df = get_rate(perc_df)
    plt.plot(
        goal_rate_df['Percentile']/100,
        goal_rate_df['Rate']/100,
        # label = curve_label,
        # color = color
    )   
       
    plt.title('Goal rate', fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('Goal Rate', fontsize=20)
    plt.xlabel('Shot Probability Model Percentile', fontsize=20)    
    plt.grid(True)


def plot_cumulative_rate(X, y, model_type):
    
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    
    probs = get_prob(X, y, model_type)
    is_goal = probs[:,1]   
    perc_df = get_percentile(X, y, probs)
    # curve_label = f[0] if len(f) == 1 else 'Distance and Angle from Net'              
        
    cumulative_rate = get_rate(perc_df, function_type='cumulative_rate') 
    
    plt.plot(
        cumulative_rate['Percentile']/100,
        cumulative_rate['Rate']/100,
        # label = curve_label,
        # color = color,
    )  
   
    plt.title('Cumulative goal rate', fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('Cumulative Proportion of Goals', fontsize=20)
    plt.xlabel('Shot Probability Model Percentile', fontsize=20)    
    plt.grid(True)


def plot_calibration(X, y, model_type):
    '''
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    '''

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2]) 

    if model_type == 'decision_tree':
        clf = DecisionTreeClassifier(    
                    max_leaf_nodes=3, 
                    max_depth=30) 

        
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.20,
                                                        random_state=50)
    y_train = y_train.values.ravel()
    clf.fit(X_train, y_train)          
        
    display = CalibrationDisplay.from_estimator(
                clf,
                X_test,
                y_test,
                n_bins=50,
                ax=ax_calibration_curve,
                # color=color,  
            )
    
    ax_calibration_curve.grid()
    plt.title("Calibration plots", fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('Fraction of positives', fontsize=20)
    plt.xlabel('Mean predicted probability', fontsize=20)
    plt.show()


def plot_models(X, y, model_type):
    
    #--------------------------------------
    # ROC
    #--------------------------------------
    plot_roc(X, y, model_type)
    plt.show()
    
    #--------------------------------------
    # Goal rate
    #--------------------------------------  
    plot_goal_rate(X, y, model_type)
    plt.show()
    
    #--------------------------------------
    # Cumulative proportion of goals
    #--------------------------------------  
    plot_cumulative_rate(X, y, model_type)
    plt.show()
    
    #--------------------------------------
    # Calibration
    #-------------------------------------- 
    plot_calibration(X, y, model_type)
    plt.show()


# if __name__ == '__main__':
    # X,y = read_dataset()