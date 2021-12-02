# coding
import os
import csv
import json
import math

# data science
import pandas as pd 
import numpy as np 


# part 2: Feature Engineering I
def q2_filter_data(file_path, dest_path, years): 
    '''
    :param file_path: string
    :param years: list
    '''   

    years = years

    columns = ['eventIdx', 'game_id', 'Distance from Net', 
                'Angle from Net', 'Is Goal', 'Is Net Empty']
    
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
        for file in os.listdir(f'{file_path}/{year}'):
            if file[-5:] == '.json':
                # Opening JSON file
                f = open(f'{file_path}/{year}/{file}')

                game_id = int(file[:-5])

                loaded_json = json.load(f)

                all_plays = loaded_json['liveData']['plays']['allPlays']

                home_team = loaded_json['gameData']['teams']['home'].get('name')

                home_side = 'NA'
                if len(loaded_json['liveData']['linescore']['periods']) > 0:
                    home_side = loaded_json['liveData']['linescore']['periods'][0]['home'].get('rinkSide')
                    
                for i in range(len(all_plays)):
                    event = all_plays[i]['result']['event']
                    if event == 'Shot' or event == 'Goal':

                        period = all_plays[i]['about']['period']

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
                        
                        

                        if x_coor and y_coor:
                            if home_side == 'right':                    
                                if home_or_away == 'Home':
                                    if int(period) % 2 == 1:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)
                                else:
                                    if int(period) % 2 == 1:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)

                            elif home_side == 'left':
                                if home_or_away == 'Home':
                                    if int(period) % 2 == 1:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)
                                        
                                else:
                                    if int(period) % 2 == 1:
                                        # shoot left
                                        distance, angle = get_distance_angle(True, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot right
                                        distance, angle = get_distance_angle(False, x_coor, y_coor)
                            
                        is_net_empty = False
                        if event == 'Goal':
                            if 'emptyNet' in all_plays[i]['result']:
                                is_net_empty = all_plays[i]['result']['emptyNet']

                        row_data = [eventIdx, game_id, distance, angle, int(event == 'Goal'), int(is_net_empty)]

                        data.append(row_data)

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