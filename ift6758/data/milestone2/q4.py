from comet_ml import Experiment
import os
import pandas as pd 

'''
Part 4 - Feature Engineering II Question 5
Filter the final dataframe and upload result to comet.ml
'''

def q5():
    all_data = pd.read_csv('/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/all_data_categorical.csv')

    the_game = all_data.loc[all_data['game_id']==2017021065]

    experiment = Experiment(
        api_key = os.environ.get("COMET_API_KEY"),
        project_name = 'milestone_2',
        workspace='xiaoxin-zhou')

    experiment.log_dataframe_profile(
        the_game, 
        name='wpg_v_wsh_2017021065',  # keep this name
        dataframe_format='csv'  # ensure you set this flag!
        )
        

q5() 