"""
Task 2:
Create a function or class to download NHL play-by-play data
 for both the regular season and playoffs.
"""
import os
import re
import json
import time
import numpy as np
import requests

def get_playoff_number():
    """
    Get all available playoff numbers. Notice that since 1st-digit is always 0, 
    this function only returns the last three digits. General naming rules:
        - 1st-digit: only 0
        - 2nd-digit in [1, 2, 3, 4] 
        - 3rd-digit bases on the 2nd:
            * if 2nd == 1: 3rd in [1, ..., 8]
            * elif 2nd == 2: 3rd in [1, ..., 4]
            * elif 2nd == 3: 3rd in [1, 2]
            * elif 2nd == 4: 3rd in [1]
        - 4st-digit in [1, ..., 7]

    Parameters:
        None

    Returns:
        playoff_numbers_dict(list): A dict of lists of 3-digit strings. 
                                    The first level of keys is the round number, or the 2nd-digit.
    """

    third_digit = fourth_digit = 0

    playoff_number_str = ""
    playoff_numbers_list = playoff_numbers_matrix = []
    playoff_numbers_dict = {
        "2nd-digit": {
            "1":[],
            "2":[],
            "3":[],
            "4":[]
        }
    }

    # Idea: ignore the first digit '0', create integers and then cast them to strings. 
    third_digit_range = [8, 4, 2, 1]

    # i+1 from 1 to 4, represents the 2nd-digit's four values: 1, 2, 3, and 4. 
    for i in range(len(third_digit_range)):

        current_limit = third_digit_range[i]

        playoff_numbers_list = []

        # 3rd and 4th digit always starts from 1
        start_number = int(str(i+1)+'11')

        # calculate the 3rd-digit range because it depends on the 2nd-digit
        end_number = start_number + current_limit*10
        
        for j in range(start_number, end_number, 10):

            # 4th-digit is independent, it always has values from 1 to 7
            current_list = list(np.arange(j, j+7, 1)) 
            playoff_numbers_list.append(current_list)

        # convert the list of lists to a Numpy array
        playoff_numbers_matrix = np.array(playoff_numbers_list)

        playoff_numbers_dict["2nd-digit"][str(i+1)] = playoff_numbers_matrix
        
    return playoff_numbers_dict

def generate_game_id(season, game_type, game_number):
    """
    Given the season of the game, the type of the game, and
    the game number, generates a 10-digit game ID. 

    Parameters:
        season(str): A 4-digit string, e.g., "2017" for the 2017-2018 season.
        game_type(str): A 2-digit string, where "01" = preseason, "02" = regular season, 
                    "03" = playoffs, "04" = all-star.
        game_number(str): A 4-digit string.
            - For regular season and preseason games, this ranges from 0001 to the number of games played. 
              (1271 for seasons with 31 teams (2017 and onwards) and 1230 for seasons with 30 teams). 

            - For playoff games, the 2nd digit of the specific number gives the round of the playoffs, 
              the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).

    Returns:
        game_id(str): A 10-digit string, e.g., "2016020001".

    Reference: https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
    """
    game_id = f"{season}{game_type}{game_number}"

    # checks if the game id is a 10-digit string.
    game_id_pattern = re.compile(r"[0-9]{10}")

    if bool(game_id_pattern.match(game_id)):
        return game_id

    else:
        print(f"Please check the inputs formats.\n \
                Current: \n\
                season - {season}\n \
                game_type - {game_type}\n \
                game_number - {game_number}")
        raise Exception

def get_data():
    """
    Download data from the 2015-16 season all the way up to the 2029-20 season.

    Parameters:
        None

    Returns:
        None
    """
    t0=time.time()
    game_id = ""

    # save JSON outputs locally
    data_local_path = "/Users/xiaoxinzhou/Documents/IFT6758_M2_JSON_data"

    years = ["2015", "2016", "2017", "2018", "2019"]

    # game types
    type_regular = "02"
    type_playoffs = "03"

    for year in years:
        # As the API mentions, there are 30 teams in 2015, and 2016
        # and 31 teams in 2017, 2018, and 2019 each.
        teams = 30 if year == "2015" or year == "2016" else 31
        
        # how many games happened
        regular_games = teams*82//2

        # ============================
        # regular season
        # ============================
        regular_data_path = f"{data_local_path}/regular_season/{year}"   

        for game in range(regular_games+1):

            # convert a number to a 4-digit string: 5 -> "0005", 123 -> "0123"
            game_number = str(game).zfill(4)

            game_id = generate_game_id(year, type_regular, game_number)
            
            response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/")
            data = response.json()

            # skip the empty game info
            if "message" in data and data["message"] == "Game data couldn't be found":
                continue

            # create file path if it doesn't exist
            filename = f"{regular_data_path}/{game_id}.json"
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            # write data to local
            with open(filename, 'w+') as f:
                json.dump(data, f, sort_keys=True, indent=4)

        # ============================
        # playoffs
        # ============================ 
        playoffs_data_path = f"{data_local_path}/playoffs/{year}"  

        # get all possible combinations of last three digits
        playoff_numbers_dict = get_playoff_number()

        # second_digit is 0, 1, 2, and 3
        for second_digit in range(4):

            two_d_nparray = playoff_numbers_dict["2nd-digit"][str(second_digit+1)]
            
            for three_digit in np.nditer(two_d_nparray):
    
                game_id = generate_game_id(year, type_playoffs, f"0{str(three_digit)}")

                response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/")
                data = response.json()

                # skip the empty game info
                if "message" in data and data["message"] == "Game data couldn't be found":
                    continue

                # create file path if it doesn't exist
                filename = f"{playoffs_data_path}/{game_id}.json"
                dirname = os.path.dirname(filename)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)

                # write data to local
                with open(filename, 'w+') as f:
                    json.dump(data, f, sort_keys=True, indent=4)
                    
                # TODO: for the purpose of discussion only, need to remove it before submit to main
                # total time: 0:01:39
    
    t1=time.time()
    import datetime 
    processing_time=str(datetime.timedelta(seconds=round(t1-t0)))
    print(f"total time: {processing_time}") 


# generate_game_id("2017", "02", "1234")
# get_playoff_number()

get_data()