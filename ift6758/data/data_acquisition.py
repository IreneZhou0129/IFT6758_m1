"""
Task 2:
Create a function or class to download NHL play-by-play data
 for both the regular season and playoffs.
"""

import re
import requests

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
              31 * 82 / 2 = 1271
            - For playoff games, the 2nd digit of the specific number gives the round of the playoffs, 
              the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).

    Returns:
        game_id(str): A 10-digit string, e.g., "2016020001".

    Reference: https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids
    """
    print("running generate_game_id()")

    game_id = f"{season}{game_type}{game_number}"

    # checks if the game id is a 10-digit string.
    game_id_pattern = re.compile(r"[0-9]{10}")

    if bool(game_id_pattern.match(game_id)):
        print(f"Game ID is: {game_id}.")
        return game_id

    else:
        print(f"Please check the inputs formats.")
        return None


def get_data():
    """
    Download data from the 2016-17 season all the way up to the 2020-21 season.
    """
    # TODO: number of teams
    teams = None
    game_id = ""

    # save JSON outputs locally
    # TODO: add the destination folder to .gitignore
    data_local_path = ""

    seasons = ["2016", "2017", "2018", "2019", "2020"]

    # game types
    regular_season = "02"
    playoffs = "03"

    # game numbers
    regular_games = teams*82/2

    # response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/{game_id}/feed/live/")
    response = requests.get(f"https://statsapi.web.nhl.com/api/v1/game/2016020001/feed/live/")

    data = response.json()    

    # TODO: save outputs by year? type of game? 

    breakpoint()

# generate_game_id("2017", "02", "1234")
get_data()