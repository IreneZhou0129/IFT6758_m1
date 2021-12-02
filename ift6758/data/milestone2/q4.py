from comet_ml import Experiment
# coding
import os
import csv
import json
import math

# data science
import pandas as pd 

def q4_categorical(file_path, dest_path, years):
    # Shot Type, Last Event Type
    years = years
    
    columns = ['eventIdx', 'game_id', 'Game Seconds', 'Game Period', 'X-Coordinate', 'Y-Coordinate',
               'Shot Distance', 'Shot Angle', 'Shot Type', 'Was Net Empty', 'Last Event Type', 'Last X-Coordinate',
               'Last Y-Coordinate', 'Time from Last Event (seconds)', 'Distance from Last Event', 'Is Rebound',
               'Change in Shot Angle', 'Speed', 'Is Goal']
    
    data = []
    
    def get_distance_angle(shoot_left, x_coor, y_coor):
        if shoot_left == True:
            distance = math.sqrt((89 + x_coor) ** 2 + (y_coor) ** 2)
            if x_coor == -89:
                if y_coor > 0:
                    angle = 0
                elif y_coor < 0:
                    angle = 180
                elif y_coor == 0:
                    angle = 90
            else:
                if y_coor > 0:
                    angle = math.degrees(math.atan((89 + x_coor) / y_coor))
                elif y_coor < 0:
                    angle = 180 - abs(math.degrees(math.atan((89 + x_coor) / y_coor)))
                elif y_coor == 0:
                    angle = 90
                    
        elif shoot_left == False:
            distance = math.sqrt((89 - x_coor) ** 2 + (y_coor) ** 2)
            if x_coor == 89:
                if y_coor > 0:
                    angle = 180
                elif y_coor < 0:
                    angle = 0
                elif y_coor == 0:
                    angle = 90
            else:
                if y_coor > 0:
                    angle = 180 - abs(math.degrees(math.atan((89 - x_coor) / y_coor)))
                elif y_coor < 0:
                    angle = abs(math.degrees(math.atan((89 - x_coor) / y_coor)))
                elif y_coor == 0:
                    angle = 90
                
        return distance, angle
    
    for year in years:
        count_jsons = 0
        for file in os.listdir(f'{file_path}/{year}'):
            if file[-5:] == '.json':
                count_jsons += 1
                # Opening JSON file
                f = open(f'{file_path}/{year}/{file}')

                game_id = int(file[:-5])

                loaded_json = json.load(f)

                all_plays = loaded_json['liveData']['plays']['allPlays']

                home_team = loaded_json['gameData']['teams']['home'].get('name')

                home_side = 'NA'
                if len(loaded_json['liveData']['linescore']['periods']) > 0:
                    home_side = loaded_json['liveData']['linescore']['periods'][0]['home'].get('rinkSide')
                    
                angle = 0
                
                for i in range(len(all_plays)):
                    event = all_plays[i]['result']['event']
                    if event == 'Shot' or event == 'Goal':
                        
                        # Q4.1 ################################################################################
                        
                        game_period = all_plays[i]['about']['period']
                        
                        period_time_mins, period_time_secs = all_plays[i]['about']['periodTime'].split(':')
                        game_seconds = ((int(game_period) - 1) * 1200) + (int(period_time_mins) * 60 + int(period_time_secs))
                        
                        team_name = all_plays[i]['team']['name']

                        if home_team == team_name:
                            home_or_away = 'Home'
                        else:
                            home_or_away = 'Away'
                        
                        eventIdx = all_plays[i]['about']['eventIdx']
                            
                        x_coor = 'NA'
                        y_coor = 'NA'
                        if 'coordinates' in all_plays[i]:
                            x_coor = all_plays[i]['coordinates'].get('x')
                            y_coor = all_plays[i]['coordinates'].get('y')
                    
                        x_coor = int(float(x_coor)) if bool(x_coor) and x_coor != 'NA' else False
                        y_coor = int(float(y_coor)) if bool(y_coor) and y_coor != 'NA' else False
                        
                        prev_angle = angle

                        if x_coor and y_coor:
                            if home_side == 'right':                    
                                if home_or_away == 'Home':
                                    if int(game_period) % 2 == 1:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)

                                    elif int(game_period) % 2 == 0:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)
                                else:
                                    if int(game_period) % 2 == 1:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)

                                    elif int(game_period) % 2 == 0:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)

                            elif home_side == 'left':
                                if home_or_away == 'Home':
                                    if int(game_period) % 2 == 1:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)

                                    elif int(game_period) % 2 == 0:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)
                                        
                                else:
                                    if int(game_period) % 2 == 1:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)

                                    elif int(game_period) % 2 == 0:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)
                            
                        
                        
                        shot_type = 'NA'
                        if 'secondaryType' in all_plays[i]['result']:
                            shot_type = all_plays[i]['result']['secondaryType']
                        
                        is_net_empty = False
                        if event == 'Goal':
                            if 'emptyNet' in all_plays[i]['result']:
                                is_net_empty = all_plays[i]['result']['emptyNet']
                                
                        # Q4.2 ################################################################################
                        
                        last_event = all_plays[i - 1]['result']['event']
                        
                        last_x_coor = 'NA'
                        last_y_coor = 'NA'
                        if 'coordinates' in all_plays[i - 1]:
                            last_x_coor = all_plays[i - 1]['coordinates'].get('x')
                            last_y_coor = all_plays[i - 1]['coordinates'].get('y')
                    
                        last_x_coor = int(float(last_x_coor)) if bool(last_x_coor) and last_x_coor != 'NA' else False
                        last_y_coor = int(float(last_y_coor)) if bool(last_y_coor) and last_y_coor != 'NA' else False
                        
                        last_game_period = all_plays[i - 1]['about']['period']
                        
                        last_period_time_mins, last_period_time_secs = all_plays[i - 1]['about']['periodTime'].split(':')
                        last_event_game_seconds = ((int(last_game_period) - 1) * 1200) + (int(last_period_time_mins) * 60 + int(last_period_time_secs))
                        time_from_last_event = game_seconds - last_event_game_seconds
                        
                        distance_from_last_event = math.sqrt(((x_coor - last_x_coor) ** 2) + ((y_coor - last_y_coor) ** 2))
                        
                        # Q4.3 ################################################################################
                        
                        is_rebound = True if last_event == 'Shot' and game_period == all_plays[i - 1]['about']['period'] else False
                        
                        change_in_shot_angle = angle - prev_angle if is_rebound else 0
                        
                        speed = distance_from_last_event / time_from_last_event if time_from_last_event != 0 else 0
                        
                        #######################################################################################

                        row_data = [eventIdx, game_id, game_seconds, game_period, x_coor, y_coor, 
                                    distance, angle, shot_type, int(is_net_empty), last_event, last_x_coor,
                                    last_y_coor, time_from_last_event, distance_from_last_event, is_rebound, 
                                    change_in_shot_angle, speed, int(event == 'Goal')]

                        if type(x_coor) == int and type(y_coor) == int and type(last_x_coor) == int and type(last_y_coor) == int:
                            data.append(row_data)

    df = pd.DataFrame.from_records(data)
    
    unique_shot_types = df.iloc[:, 8].unique()
    print(unique_shot_types)
    df.iloc[:, 8].replace(to_replace=unique_shot_types,
           value=list(range(len(unique_shot_types))),
           inplace=True)
    
    unique_last_event_types = df.iloc[:, 10].unique()
    print(unique_last_event_types)
    df.iloc[:, 10].replace(to_replace=unique_last_event_types,
           value=list(range(len(unique_last_event_types))),
           inplace=True)
    
    print(df.dtypes)
        
    data = df.values.tolist()
    
    # create file path if it doesn't exist
    filename = f'{dest_path}'
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # writing to csv file 
    with open(filename, 'w+') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 

        # writing the fields 
        csvwriter.writerow(columns) 

        # writing the data rows 
        csvwriter.writerows(data)    

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