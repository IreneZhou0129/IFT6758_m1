# coding
import os
import csv
import json
import math

# data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
                    angle = 180 - \
                        abs(math.degrees(math.atan((89 + x_coor) / y_coor)))
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
                    angle = 180 - \
                        abs(math.degrees(math.atan((89 - x_coor) / y_coor)))
                elif y_coor < 0:
                    angle = abs(math.degrees(
                        math.atan((89 - x_coor) / y_coor)))
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

                home_team = loaded_json['gameData']['teams']['home'].get(
                    'name')

                home_side = 'NA'
                if len(loaded_json['liveData']['linescore']['periods']) > 0:
                    home_side = loaded_json['liveData']['linescore']['periods'][0]['home'].get(
                        'rinkSide')

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

                        x_coor = int(float(x_coor)) if bool(
                            x_coor) and x_coor != 'NA' else False
                        y_coor = int(float(y_coor)) if bool(
                            y_coor) and y_coor != 'NA' else False

                        if x_coor and y_coor:
                            if home_side == 'right':
                                if home_or_away == 'Home':
                                    if int(period) % 2 == 1:
                                        # shoot left
                                        distance, angle = get_distance_angle(
                                            True, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot right
                                        distance, angle = get_distance_angle(
                                            False, x_coor, y_coor)
                                else:
                                    if int(period) % 2 == 1:
                                        # shoot right
                                        distance, angle = get_distance_angle(
                                            False, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot left
                                        distance, angle = get_distance_angle(
                                            True, x_coor, y_coor)

                            elif home_side == 'left':
                                if home_or_away == 'Home':
                                    if int(period) % 2 == 1:
                                        # shoot right
                                        distance, angle = get_distance_angle(
                                            False, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot left
                                        distance, angle = get_distance_angle(
                                            True, x_coor, y_coor)

                                else:
                                    if int(period) % 2 == 1:
                                        # shoot left
                                        distance, angle = get_distance_angle(
                                            True, x_coor, y_coor)

                                    elif int(period) % 2 == 0:
                                        # shoot right
                                        distance, angle = get_distance_angle(
                                            False, x_coor, y_coor)

                        is_net_empty = False
                        if event == 'Goal':
                            if 'emptyNet' in all_plays[i]['result']:
                                is_net_empty = all_plays[i]['result']['emptyNet']

                        row_data = [eventIdx, game_id, distance, angle, int(
                            event == 'Goal'), int(is_net_empty)]

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


def q2_1_1(csv_path):
    shot_distances = []
    goal_distances = []

    # csv_path = f'../../m2_CSV_data/all_data.csv'
    csv_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/all_data.csv'
    with open(csv_path) as csvfile:
        data = csv.DictReader(csvfile)

        for row in data:
            is_goal = int(row.get('Is Goal'))
            if is_goal == 1:
                goal_distances.append(float(row.get('Distance from Net')))
            elif is_goal == 0:
                shot_distances.append(float(row.get('Distance from Net')))

    manual_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                   105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185,
                   190]
    manual_bins_string = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55',
                          '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100', '100-105',
                          '105-110', '110-115', '115-120', '120-125', '125-130', '130-135', '135-140', '140-145',
                          '145-150', '150-155', '155-160', '160-165', '165-170', '170-175', '175-180', '180-185', '185-190']

    (shot_counts, shot_bins, patches) = plt.hist(
        shot_distances, bins=manual_bins)
    print(shot_counts)
    print(shot_bins)
    (goal_counts, goal_bins, patches) = plt.hist(goal_distances, bins=shot_bins)
    print(goal_counts)
    plt.close()

    plt.figure(figsize=(45, 8))
    plt.bar(manual_bins_string, shot_counts)
    plt.bar(manual_bins_string, goal_counts, bottom=shot_counts)
    plt.xlabel("Shot Distance (feet)", fontsize=20)
    plt.ylabel("Shot or Goal Count", fontsize=20)
    plt.legend(["Shots", "Goals"], loc=2, prop={'size': 16})
    plt.title(
        f"Shot or Goal Counts based on Shot Distance in the 2015-2018 Season", fontsize=20)

    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=16)

    plt.show()
    plt.close()


def q2_1_2(csv_path):
    shot_angles = []
    goal_angles = []

    # csv_path = f'../../m2_CSV_data/all_data.csv'

    with open(csv_path) as csvfile:
        data = csv.DictReader(csvfile)

        for row in data:
            is_goal = int(row.get('Is Goal'))
            if is_goal == 1:
                goal_angles.append(float(row.get('Angle from Net')))
            elif is_goal == 0:
                shot_angles.append(float(row.get('Angle from Net')))

    manual_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,
                   105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

    manual_bins_string = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55',
                          '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100', '100-105',
                          '105-110', '110-115', '115-120', '120-125', '125-130', '130-135', '135-140', '140-145',
                          '145-150', '150-155', '155-160', '160-165', '165-170', '170-175', '175-180']

    (shot_counts, shot_bins, patches) = plt.hist(shot_angles, bins=manual_bins)
    # print(shot_counts)
    # print(shot_bins)
    (goal_counts, goal_bins, patches) = plt.hist(goal_angles, bins=shot_bins)
    # print(goal_counts)
    plt.close()

    plt.figure(figsize=(45, 8))
    plt.bar(manual_bins_string, shot_counts)
    plt.bar(manual_bins_string, goal_counts, bottom=shot_counts)
    plt.xlabel("Shot Angle (degrees)", fontsize=20)
    plt.ylabel("Shot or Goal Count", fontsize=20)
    plt.legend(["Shots", "Goals"], loc=2, prop={'size': 16})
    plt.title(
        f"Shot or Goal Counts based on Shot Angle in the 2015-2018 Season", fontsize=20)

    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=16)

    plt.show()
    plt.close()

# histogram


def q2_1_histogram(csv_path):

    all_data = pd.read_csv(csv_path)

    # sns.jointplot(
    #     data=all_data, 
    #     x="Distance from Net", 
    #     y="Angle from Net", 
    #     kind='hist', 
    #     height=30)
    sns.jointplot(
        data=all_data, 
        x="Distance from Net", 
        y="Angle from Net", 
        kind='hist', 
        height=12)
    
    plt.xlabel("Distance from Net", fontsize=20)
    plt.ylabel("Angle from Net", fontsize=20)
    # plt.title('Histogram of shot counts', fontsize=20)
    plt.show()

def q2_2_1(csv_path):
    shot_distances = []
    goal_distances = []
                        
    with open(csv_path) as csvfile:
        data = csv.DictReader(csvfile)

        for row in data:
            is_goal = int(row.get('Is Goal'))
            if is_goal == 1:
                goal_distances.append(float(row.get('Distance from Net')))
            elif is_goal == 0:
                shot_distances.append(float(row.get('Distance from Net')))
                    
    manual_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                   105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 
                   190]
    
    manual_bins_string = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55',
             '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100', '100-105', 
             '105-110', '110-115', '115-120', '120-125', '125-130', '130-135', '135-140', '140-145', 
             '145-150', '150-155', '155-160', '160-165', '165-170', '170-175', '175-180', '180-185', '185-190']
    
    (shot_counts, shot_bins, patches) = plt.hist(shot_distances, bins=manual_bins)
    print(shot_counts)
    print(shot_bins)
    (goal_counts, goal_bins, patches) = plt.hist(goal_distances, bins=shot_bins)
    print(goal_counts)
    plt.close()
    
    goal_chance = goal_counts / (goal_counts + shot_counts)
    print(goal_chance)
    
    plt.figure(figsize=(45, 8))
    
    plt.plot(manual_bins_string, goal_chance)
    plt.xlabel("Shot Distance (feet)", fontsize=20)
    plt.ylabel("Goal Percentage", fontsize=20)
    plt.title(f"Goal Percentage based on Shot Distance in the 2015-2018 Season", fontsize=20)

    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=16)

    plt.show()

def q2_2_2(csv_path):
    shot_angles = []
    goal_angles = []
                        
    with open(csv_path) as csvfile:
        data = csv.DictReader(csvfile)

        for row in data:
            is_goal = int(row.get('Is Goal'))
            if is_goal == 1:
                goal_angles.append(float(row.get('Angle from Net')))
            elif is_goal == 0:
                shot_angles.append(float(row.get('Angle from Net')))
    
    manual_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                   105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]
    
    manual_bins_string = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', 
                          '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100', '100-105', 
                          '105-110', '110-115', '115-120', '120-125', '125-130', '130-135', '135-140', '140-145', 
                          '145-150', '150-155', '155-160', '160-165', '165-170', '170-175', '175-180']
    
    (shot_counts, shot_bins, patches) = plt.hist(shot_angles, bins=manual_bins)
    print(shot_counts)
    print(shot_bins)
    (goal_counts, goal_bins, patches) = plt.hist(goal_angles, bins=shot_bins)
    print(goal_counts)
    plt.close()
    
    goal_chance = goal_counts / (goal_counts + shot_counts)
    print(goal_chance)
    
    plt.figure(figsize=(45, 8))
    # plt.bar(manual_bins_string, goal_chance)
    plt.plot(manual_bins_string, goal_chance)
    plt.xlabel("Shot Angle (degrees)", fontsize=20)
    plt.ylabel("Goal Percentage", fontsize=20)
    plt.title(f"Goal Percentage based on Shot Angle in the 2015-2018 Season", fontsize=20)

    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=16)

    plt.show()

def q2_3(csv_path):
    empty_net = []
    non_empty_net = []
                    
    with open(csv_path) as csvfile:
        data = csv.DictReader(csvfile)

        for row in data:
            if int(row.get('Is Goal')) == 1:
                if int(row.get('Is Net Empty')) == 1:
                    empty_net.append(float(row.get('Distance from Net')))
                else:
                    non_empty_net.append(float(row.get('Distance from Net')))
                    
    manual_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 
                   105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 
                   190]
    
    manual_bins_string = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55',
             '55-60', '60-65', '65-70', '70-75', '75-80', '80-85', '85-90', '90-95', '95-100', '100-105', 
             '105-110', '110-115', '115-120', '120-125', '125-130', '130-135', '135-140', '140-145', 
             '145-150', '150-155', '155-160', '160-165', '165-170', '170-175', '175-180', '180-185', '185-190']
    
    (empty_net_counts, goal_bins, patches) = plt.hist(empty_net, bins=manual_bins)
    print(empty_net_counts)
    print(goal_bins)
    (non_empty_net_counts, goal_bins, patches) = plt.hist(non_empty_net, bins=goal_bins)
    print(non_empty_net_counts)
    plt.close()
    
    plt.figure(figsize=(45, 8))

    plt.bar(manual_bins_string, non_empty_net_counts)
    plt.bar(manual_bins_string, empty_net_counts, bottom=non_empty_net_counts)
    plt.xlabel("Goal Distance (feet)", fontsize=20)
    plt.ylabel("Empty or Non-empty Net Count", fontsize=20)
    plt.legend(["Non-empty Net", "Empty Net"], loc=2, prop={'size': 16})
    plt.title(f"Empty or Non-empty Net Counts based on Shot Distance in the 2015-2018 Season", fontsize=20)
    
    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=16)

    plt.show()
    plt.close()

if __name__ == '__main__':
    # csv_path = f'../../m2_CSV_data/all_data.csv'
    csv_path = '/Users/xiaoxinzhou/Documents/IFT6758_M2_CSV_data/all_data.csv'

    # q2_1_1(csv_path)
    # q2_1_2(csv_path)
    # q2_1_histogram(csv_path)

    # q2_2_1(csv_path)
    # q2_2_2(csv_path) 

    q2_3(csv_path)