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

    # Path('models/').mkdir(exist_ok=True)
    

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

    # save the model to disk
    filename = 'models\log_reg\log_reg_dist.pkl'
    pickle.dump(clf, open(filename, 'wb'))

    experiment.log_model('logreg_model', filename)

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


  


def get_prob(X, y, model_type, features=['Distance from Net']):
    '''
     Calculate the probability.
     In X_test_pred_proba, the first column is label 0, the second one is label 1.

     :param model_type: Logistic regression or XGBoost
    '''
    # Create a training and validation split
    X_train, X_test, y_train, y_test = train_test_split(X[features],y,test_size=0.20,random_state=50)

    if model_type == 'logreg':
        # Logistic regression model fitting
        clf = LogisticRegression()
        y_train = y_train.values.ravel()
    
    elif model_type == 'xgb':
        # Fit model no training data
        clf = xgb.XGBClassifier()
    
    elif model_type == 'decision_tree':
        clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)  

    clf.fit(X_train, y_train)

    # Predict the probability
    X_test_pred_proba = clf.predict_proba(X_test)

    # print(f'The probabilities on validation set is\n {X_test_pred_proba}')
    
    return X_test_pred_proba

def get_percentile(X, y, probs, features=['Distance from Net']):
    '''
    Return a df that has four columns:
        * 'index': index in y_test.
        * 'Goal prob': Goal probability.
        * 'Is Goal': 0 means not goal; 1 means goal.
        * 'Percentile': Calculated percentile, range from 0.00 to 99.99.
    The df is sorted by 'Percentile'.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X[features],y,test_size=0.20,random_state=50)

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
def plot_roc(X, y, feature_color_dict, model_type):

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])    

    for k,v in feature_color_dict.items():
        
        f = v[0]
        color = v[1]
        
        if f != 'Random baseline':
            X_train, X_test, y_train, y_test = train_test_split(X[f],y,test_size=0.20,random_state=50)
            probs = get_prob(X, y, model_type, f)
            is_goal = probs[:,1]                

        # Random baseline
        else:
            is_goal = np.random.uniform(0,1,is_goal.shape[0])
        
        fpr, tpr, threshold = metrics.roc_curve(y_test, is_goal)
        
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.plot(
            fpr, 
            tpr, 
            color = color, 
            label = f'{f} '+'AUC = %0.2f' % roc_auc)      
    
    plt.axis([0, 1, 0, 1])    
        
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)


def plot_goal_rate(X, y, feature_color_dict, model_type):
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])    

    for k,v in feature_color_dict.items():
        
        f = v[0]
        color = v[1]
        
        if f != 'Random baseline':
            probs = get_prob(X, y, model_type, f)
            is_goal = probs[:,1]   
            perc_df = get_percentile(X, y, probs, f)

        # Random baseline
        else:
            is_goal = np.random.uniform(0,1,is_goal.shape[0])
            no_goal_prob = np.array([(1-i) for i in is_goal])
            probs = np.column_stack((is_goal, no_goal_prob))
            perc_df = get_percentile(X, y, probs)
        
        goal_rate_df = get_rate(perc_df)
        plt.plot(
            goal_rate_df['Percentile']/100,
            goal_rate_df['Rate']/100,
            label = f'{f}',
            color = color
        )   
       
    plt.title('Goal rate')
    plt.legend()
    plt.grid(True)


def plot_cumulative_rate(X, y, feature_color_dict, model_type):
    
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    
    for k,v in feature_color_dict.items():
        
        f = v[0]
        color = v[1]

        if f != 'Random baseline':
            probs = get_prob(X, y, model_type, f)
            is_goal = probs[:,1]   
            perc_df = get_percentile(X, y, probs, f)

        # Random baseline
        else:
            is_goal = np.random.uniform(0,1,is_goal.shape[0])
            no_goal_prob = np.array([(1-i) for i in is_goal])
            probs = np.column_stack((is_goal, no_goal_prob))
            perc_df = get_percentile(X, y, probs)
            
        cumulative_rate = get_rate(perc_df, function_type='cumulative_rate') 
        
        plt.plot(
            cumulative_rate['Percentile']/100,
            cumulative_rate['Rate']/100,
            label = f'{f}',
            color = color,
        )  
   
    plt.title('Cumulative goal rate')
    plt.legend()
    plt.grid(True)


def plot_calibration(X, y, feature_color_dict, model_type):
    '''
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    '''

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])

    # select model
    if model_type == 'logreg':
        # Logistic regression model fitting
        clf = LogisticRegression()
    
    elif model_type == 'xgb':
        # Fit model no training data
        clf = xgb.XGBClassifier()  

    elif model_type == 'decision_tree':
        clf = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)

    for k,v in feature_color_dict.items():
        
        f = v[0]
        color = v[1]
        
        if f != 'Random baseline':
            X_train, X_test, y_train, y_test = train_test_split(X[f],
                                                                y,
                                                                test_size=0.20,
                                                                random_state=50)
            y_train = y_train.values.ravel()
            clf.fit(X_train, y_train)
        
        # Random baseline
        else:
            goal_prob = np.random.uniform(0, 1, X_test.shape[0])
            
            # Value is 1 if goal_prob is greater than 0.5, 0 otherwise.
            random_y_test = np.zeros((goal_prob.shape[0],1))
            random_y_test[:,][np.where(goal_prob>0.5)]=1
        
        display = CalibrationDisplay.from_estimator(
                clf,
                X_test,
                y_test,
                n_bins=50,
                ax=ax_calibration_curve,
                color=color,
                label=f'{f}'
            )
    
    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots (SVC)")

    plt.show()


def plot_models(X, y, model_type, features=['Distance from Net']):
    '''
    :param features: A list. Possible items are 'Distance from Net', 'Angle from Net', and both.    
    :param model_type: 'logreg' or 'xgb'
    '''

    feature_color_dict = {
        1: [['Distance from Net'], 'r'],
        2: [['Angle from Net'], 'g'],
        3: [['Distance from Net', 'Angle from Net'], 'plum'],
        4: ['Random baseline', 'b']
        }
    
    #--------------------------------------
    # ROC
    #--------------------------------------
    plot_roc(X, y, feature_color_dict, model_type)
    plt.show()
    
    #--------------------------------------
    # Goal rate
    #--------------------------------------  
    plot_goal_rate(X, y, feature_color_dict, model_type)
    plt.show()
    
    #--------------------------------------
    # Cumulative proportion of goals
    #--------------------------------------  
    plot_cumulative_rate(X, y, feature_color_dict, model_type)
    plt.show()
    
    #--------------------------------------
    # Calibration
    #-------------------------------------- 
    plot_calibration(X, y, feature_color_dict, model_type)
    plt.show()


if __name__ == '__main__':
    X,y = read_dataset()
    train(X, y)


    
    # train('distance_and_angle', 
    #         'models/distance_and_angle.h5', 
    #         features=['Distance from Net', 'Angle from Net'])      