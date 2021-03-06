# coding
import pickle

# data science
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.calibration import calibration_curve, CalibrationDisplay

# customized APIs
from q6_baseline import get_prob, \
                        get_percentile, \
                        get_rate, \
                        get_clf, \
                        read_all_features

def plot_roc(X, y, five_curves,  q2_path, q4_path,season_type):

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])    

    for k,v in five_curves.items():
        print(f'*************************\nrunning {k}\n*************************')

        X, y = read_all_features(v['data_path'])

        model_type = v['model_type']

        # find right X
        if v['all_feature'] == False:
            X = X[v['feature']]        
        
        color = v['color']
        curve_label = v['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=50)
        probs = get_prob(X, y, model_type)
        is_goal = probs[:,1]              

        fpr, tpr, threshold = metrics.roc_curve(y_test, is_goal)
        
        roc_auc = metrics.auc(fpr, tpr)
        
        plt.plot(
            fpr, 
            tpr, 
            color = color, 
            label = f'{curve_label} '+'AUC = %0.2f' % roc_auc)      
    
    plt.axis([0, 1, 0, 1])    
        
    plt.title(f'ROC Curves ({season_type})', fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=20)    
    plt.grid(True)
    plt.show()

def plot_goal_rate(X, y, five_curves, q2_path, q4_path,season_type):
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])    

    for k,v in five_curves.items():
        print(f'*************************\nrunning {k}\n*************************')
       
        X, y = read_all_features(v['data_path'])

        model_type = v['model_type']

        # find right X
        if v['all_feature'] == False:
            X = X[v['feature']]        
        
        color = v['color']
        curve_label = v['label']
        
        probs = get_prob(X, y, model_type)
        is_goal = probs[:,1]   
        perc_df = get_percentile(X, y, probs)
        # curve_label = f[0] if len(f) == 1 else 'Distance and Angle from Net'              

        goal_rate_df = get_rate(perc_df)
        plt.plot(
            goal_rate_df['Percentile']/100,
            goal_rate_df['Rate']/100,
            label = curve_label,
            color = color
        )   
        
    plt.title(f'Goal rate ({season_type})', fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('Goal Rate', fontsize=20)
    plt.xlabel('Shot Probability Model Percentile', fontsize=20)    
    plt.grid(True)
    plt.show()

def plot_cumulative_rate(X, y, five_curves, q2_path, q4_path,season_type):
    
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    
    for k,v in five_curves.items():
        print(f'*************************\nrunning {k}\n*************************')    
        
        X, y = read_all_features(v['data_path'])

        model_type = v['model_type']

        # find right X
        if v['all_feature'] == False:
            X = X[v['feature']]        
        
        color = v['color']
        curve_label = v['label']

        probs = get_prob(X, y, model_type)
        is_goal = probs[:,1]   
        perc_df = get_percentile(X, y, probs)
        # curve_label = f[0] if len(f) == 1 else 'Distance and Angle from Net'              
            
        cumulative_rate = get_rate(perc_df, function_type='cumulative_rate') 
        
        plt.plot(
            cumulative_rate['Percentile']/100,
            cumulative_rate['Rate']/100,
            label = curve_label,
            color = color,
        )  
   
    plt.title(f'Cumulative goal rate ({season_type})', fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('Cumulative Proportion of Goals', fontsize=20)
    plt.xlabel('Shot Probability Model Percentile', fontsize=20)    
    plt.grid(True)
    plt.show()

def plot_calibration(X, y, five_curves, q2_path, q4_path,season_type):
    '''
    https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    '''

    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2)

    ax_calibration_curve = fig.add_subplot(gs[:2, :2]) 

    for k,v in five_curves.items():
        print(f'*************************\nrunning {k}\n*************************')

        X, y = read_all_features(v['data_path'])

        model_type = v['model_type']

        # find right X
        if v['all_feature'] == False:
            X = X[v['feature']]        
        
        color = v['color']
        curve_label = v['label']

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.20,
                                                            random_state=50)
        clf = get_clf(X_train, X_test,model_type,y_train)   

        y_train = y_train.values.ravel()
        clf.fit(X_train, y_train)          
            
        display = CalibrationDisplay.from_estimator(
                    clf,
                    X_test,
                    y_test,
                    n_bins=50,
                    ax=ax_calibration_curve,
                    label = curve_label,
                    color = color,
            )
    
    ax_calibration_curve.grid()
    plt.title(f"Calibration plots ({season_type})", fontsize=20)
    plt.legend(loc=2,prop={'size': 16})
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    plt.ylabel('Fraction of positives', fontsize=20)
    plt.xlabel('Mean predicted probability', fontsize=20)
    plt.show()    

if __name__=='__main__':
    q2_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_simple.csv'
    q4_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_categorical.csv'
    
    X, y = read_all_features()

    five_curves = {
        '1':{
            'pickle_name': 'models\log_reg\log_reg_dist.pkl',
            'all_feature': False,
            'feature': ['Distance from Net'],
            'model_type': 'logreg',
            'data_path': q2_path,
            'color': 'r',
            'label': 'Distance from Net'
        },
        '2':{
            'pickle_name': 'models\log_reg\log_reg_angle.pkl',
            'all_feature': False,
            'feature': ['Angle from Net'],
            'model_type': 'logreg',
            'data_path': q2_path,
            'color': 'g',
            'label': 'Angle from Net'    
        },
        '3':{
            'pickle_name': 'models\log_reg\log_reg_both.pkl',
            'all_feature': False,
            'feature': ['Distance from Net', 'Angle from Net'],
            'model_type': 'logreg',
            'data_path': q2_path,
            'color': 'b', 
            'label': 'Distance and Angle from Net'             
        },
        '4':{
            'pickle_name': 'q5_3_selectKBest.pkl',
            'all_feature': True,
            'model_type': 'xgb_tech_2',
            'data_path': q4_path,
            'color': 'r',
            'label': 'XGBoost-selectKBest'
        },
        '5':{
            'pickle_name': 'decision_tree/approach_4.pkl',
            'all_feature': True,
            'model_type': 'approach_4',
            'data_path': q4_path,
            'color': 'plum', 
            'label': 'MLP'              
        }
    }

    

    # q2_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_simple.csv'
    # q4_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_categorical.csv'

    q2_path_playoff = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_simple_playoffs.csv'
    q4_path_playoff = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/test_data_categorical_playoffs.csv'

    season_type = '2019/20 playoffs test set'

    q2_path = q2_path_playoff
    q4_path = q4_path_playoff

    plot_roc(X, y, five_curves, q2_path, q4_path,season_type)
    plot_goal_rate(X, y, five_curves, q2_path, q4_path,season_type)
    plot_cumulative_rate(X, y, five_curves, q2_path, q4_path,season_type)
    plot_calibration(X, y, five_curves, q2_path, q4_path,season_type)

    # 7 plots (regular; playoffs)
    # 1. different DF
    # 2. 
