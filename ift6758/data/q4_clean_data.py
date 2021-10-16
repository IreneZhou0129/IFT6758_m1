import json
import pprint
import pandas as pd
import csv
import os


def q4():
    years = [2016, 2017, 2018, 2019, 2020]
    game_types = ['regular_season', 'playoffs']
    columns = ['eventIdx', 'Date & Time', 'Period', 'Period Time', 'Period Time Remaining', 'Period Type', 
               'Game ID', 'Team Name', 'Home or Away', 'Shot or Goal', 'X-Coordinate', 'Y-Coordinate', 
               'Shooter Name', 'Goalie Name', 'Scorer Name', 'Shot Type', 'Was Net Empty', 
               'Goalie Strength', 'Home Rink Side']

    for year in years:
        for game_type in game_types:
            for file in os.listdir(f'/Users/xiaoxinzhou/Documents/IFT6758_JSON_data/{game_type}/{year}'):
                if file[-5:] == '.json':
                    # Opening JSON file
                    f = open(f'/Users/xiaoxinzhou/Documents/IFT6758_JSON_data/{game_type}/{year}/{file}')

                    game_id = int(file[:-5])

                    loaded_json = json.load(f)

                    all_plays = loaded_json['liveData']['plays']['allPlays']

                    data = []
                    
                    home_team = loaded_json['gameData']['teams']['home'].get('name')
                    
                    home_side = 'NA'
                    if len(loaded_json['liveData']['linescore']['periods']) > 0:
                        home_side = loaded_json['liveData']['linescore']['periods'][0]['home'].get('rinkSide')

                    for i in range(len(all_plays)):
                        event = all_plays[i]['result']['event']
                        if event == 'Shot' or event == 'Goal':
                            # pprint.pprint(all_plays[i]['result'])    
                            date_time = all_plays[i]['about']['dateTime']

                            period = all_plays[i]['about']['period']
                            period_time = all_plays[i]['about']['periodTime']
                            period_time_remaining = all_plays[i]['about']['periodTimeRemaining']
                            period_type = all_plays[i]['about']['periodType']

                            team_name = all_plays[i]['team']['name']
                            
                            if home_team == team_name:
                                home_or_away = 'Home'
                            else:
                                home_or_away = 'Away'

                            eventIdx = all_plays[i]['about']['eventIdx']

                            coord_x = 'NA'
                            coord_y = 'NA'
                            if 'coordinates' in all_plays[i]:
                                coord_x = all_plays[i]['coordinates'].get('x')
                                coord_y = all_plays[i]['coordinates'].get('y')

                            scorer_name = 'NA'
                            for player in all_plays[i]['players']:
                                if player['playerType'] == 'Shooter':
                                    shooter_name = player['player']['fullName']
                                elif player['playerType'] == 'Goalie':
                                    goalie_name = player['player']['fullName']
                                elif player['playerType'] == 'Scorer':
                                    scorer_name = player['player']['fullName']   

                            shot_type = 'NA'
                            if 'secondaryType' in all_plays[i]['result']:
                                shot_type = all_plays[i]['result']['secondaryType']

                            was_net_empty = False
                            goalie_strength = 'NA'
                            if event == 'Goal':
                                if 'emptyNet' in all_plays[i]['result']:
                                    was_net_empty = all_plays[i]['result']['emptyNet']

                                goalie_strength = all_plays[i]['result']['strength']['name']   

                            row_data = [eventIdx, date_time, period, period_time, period_time_remaining, 
                                        period_type, game_id, team_name, home_or_away, event, coord_x, coord_y, 
                                        shooter_name, goalie_name, scorer_name, shot_type, 
                                        was_net_empty, goalie_strength, home_side]

                            data.append(row_data)

                # create file path if it doesn't exist
                filename = f'/Users/xiaoxinzhou/Documents/IFT6758_CSV_data/{game_type}/{year}/{game_id}.csv'
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


                """
                For each event, you will want to include as features (at minimum): 
                game time/period information, 
                game ID, 
                team information (which team took the shot), 
                indicator if its a shot or a goal, 
                the on-ice coordinates, 
                the shooter and goalie name (don’t worry about assists for now), 
                shot type, 
                if it was on an empty net, 
                and whether or not a goal was at even strength, 
                shorthanded, 
                or on the power play.
                """
                
q4()