import numpy as np
import csv
import json
import os
import pandas as pd
import plotly.offline as pyo
import plotly.express as px  # (version 4.7.0 or higher)
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

def get_league_average_table(year):
    
    csv_path = f'/Users/xiaoxinzhou/Documents/IFT6758_CSV_data/regular_season/{year}'

    league_total_table = np.zeros((85, 200))
    
    for filename in os.listdir(csv_path):
        # df = pd.read_csv(f'{csv_path}/{filename}')

        with open(csv_path + '/' + filename) as csvfile:
            data = csv.DictReader(csvfile)
            
            #open corresponding json file and get start home court side
            
            f = open(f'/Users/xiaoxinzhou/Documents/IFT6758_JSON_data/regular_season/{year}/{filename[:-4]}.json')
            loaded_json = json.load(f)
            home_side = loaded_json['liveData']['linescore']['periods'][0]['home'].get('rinkSide')
                    
            for row in data:

                y_coor = row.get('Y-Coordinate') 
                x_coor = row.get('X-Coordinate')

                y_coor = int(float(y_coor)) if bool(y_coor) and y_coor != 'NA' else False
                x_coor = int(float(x_coor)) if bool(x_coor) and x_coor != 'NA' else False

                if x_coor and y_coor:
                    if home_side == 'right':                    
                        if row.get('Home or Away') == 'Home':
                            if int(row.get('Period')) % 2 == 1:
                                # print(row.get('Y-Coordinate'))
                                # shoot left
                                league_total_table[42 - y_coor, 100 + x_coor] += 1

                            elif int(row.get('Period')) % 2 == 0:
                                # shoot right
                                league_total_table[42 + y_coor, 100 - x_coor] += 1
                        else:
                            if int(row.get('Period')) % 2 == 1:
                                # shoot right
                                league_total_table[42 + y_coor, 100 - x_coor] += 1

                            elif int(row.get('Period')) % 2 == 0:
                                # shoot left
                                league_total_table[42 - y_coor, 100 + x_coor] += 1
                                
                    elif home_side == 'left':
                        if row.get('Home or Away') == 'Home':
                            if int(row.get('Period')) % 2 == 1:
                                # print(row.get('Y-Coordinate'))
                                # shoot right
                                league_total_table[42 + y_coor, 100 - x_coor] += 1

                            elif int(row.get('Period')) % 2 == 0:
                                # shoot left
                                league_total_table[42 - y_coor, 100 + x_coor] += 1
                        else:
                            if int(row.get('Period')) % 2 == 1:
                                # shoot left
                                league_total_table[42 - y_coor, 100 + x_coor] += 1

                            elif int(row.get('Period')) % 2 == 0:
                                # shoot right
                                league_total_table[42 + y_coor, 100 - x_coor] += 1

    num_teams = 30 if int(year) == 2016 else 31
    league_average_table = league_total_table / num_teams
    # print(league_average_table)
    return league_average_table

def get_team_table(team, year):
    
    csv_path = f'/Users/xiaoxinzhou/Documents/IFT6758_CSV_data/regular_season/{year}'

    team_table = np.zeros((85, 200))

    for filename in os.listdir(csv_path):

        #open corresponding json file and get start home court side
        f = open(f'/Users/xiaoxinzhou/Documents/IFT6758_JSON_data/regular_season/{year}/{filename[:-4]}.json')
        loaded_json = json.load(f)
        home_side = loaded_json['liveData']['linescore']['periods'][0]['home'].get('rinkSide')
        home_team = loaded_json['gameData']['teams']['home'].get('name')
        away_team = loaded_json['gameData']['teams']['away'].get('name')
        
        if team == home_team or team == away_team:
            with open(csv_path + '/' + filename) as csvfile:
                data = csv.DictReader(csvfile)

                for row in data:
                    if row.get('Team Name') == team:
                        y_coor = row.get('Y-Coordinate')
                        x_coor = row.get('X-Coordinate')

                        y_coor = int(float(y_coor)) if bool(y_coor) and y_coor != 'NA' else False
                        x_coor = int(float(x_coor)) if bool(x_coor) and x_coor != 'NA' else False

                        if x_coor and y_coor:
                            if home_side == 'right':
                                if row.get('Home or Away') == 'Home':
                                    if int(row.get('Period')) % 2 == 1:
                                        # print(row.get('Y-Coordinate'))
                                        # shoot left
                                        team_table[42 - y_coor, 100 + x_coor] += 1

                                    elif int(row.get('Period')) % 2 == 0:
                                        # shoot right
                                        team_table[42 + y_coor, 100 - x_coor] += 1
                                else:
                                    if int(row.get('Period')) % 2 == 1:
                                        # shoot right
                                        team_table[42 + y_coor, 100 - x_coor] += 1

                                    elif int(row.get('Period')) % 2 == 0:
                                        # shoot left
                                        team_table[42 - y_coor, 100 + x_coor] += 1
                                        
                            elif home_side == 'left':
                                if row.get('Home or Away') == 'Home':
                                    if int(row.get('Period')) % 2 == 1:
                                        # print(row.get('Y-Coordinate'))
                                        # shoot right
                                        team_table[42 + y_coor, 100 - x_coor] += 1

                                    elif int(row.get('Period')) % 2 == 0:
                                        # shoot left
                                        team_table[42 - y_coor, 100 + x_coor] += 1
                                else:
                                    if int(row.get('Period')) % 2 == 1:
                                        # shoot left
                                        team_table[42 - y_coor, 100 + x_coor] += 1

                                    elif int(row.get('Period')) % 2 == 0:
                                        # shoot right
                                        team_table[42 + y_coor, 100 - x_coor] += 1

    return team_table

team_table = get_team_table('Colorado Avalanche', '2016')    

df = pd.DataFrame(team_table)

teams_set = {'Philadelphia Flyers', 'Vancouver Canucks', 'St. Louis Blues', 
            'Ottawa Senators', 'Toronto Maple Leafs', 'Vegas Golden Knights', 
            'New York Rangers', 'New York Islanders', 'New Jersey Devils', 
            'Winnipeg Jets', 'Chicago Blackhawks', 'Boston Bruins', 
            'Detroit Red Wings', 'San Jose Sharks', 'Anaheim Ducks', 
            'Florida Panthers', 'Montréal Canadiens', 'Columbus Blue Jackets', 
            'Calgary Flames', 'Nashville Predators', 'Buffalo Sabres', 
            'Edmonton Oilers', 'Tampa Bay Lightning', 'Minnesota Wild', 
            'Colorado Avalanche', 'Dallas Stars', 'Los Angeles Kings', 
            'Carolina Hurricanes', 'Arizona Coyotes', 'Washington Capitals', 
            'Pittsburgh Penguins'}
teams_list = []
for t in teams_set:
    teams_list.append(
        {'label':t, 'value': t}
    )

app = Dash(__name__)

# ------------------------------------------------------------------------------
# App layout
app.layout = html.Div([

    html.H1("Web Application Dashboards with Dash", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2016", "value": "2016"},
                     {"label": "2017", "value": "2017"},
                     {"label": "2018", "value": "2018"},
                     {"label": "2019", "value": "2019"},
                     {"label": "2020", "value": "2020"}],
                 multi=False,
                 value=2016,
                 style={'width': "40%"}
                 ),
    dcc.Dropdown(id="slct_team",
                 options=teams_list,
                 multi=False,
                 value='Philadelphia Flyers',
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={})

])

# ------------------------------------------------------------------------------
# Connect the Plotly graphs with Dash Components
@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value'),
    Input(component_id='slct_team', component_property='value')]
)
def update_graph(option_slctd_year,option_slctd_team):

    year_container = f""

    dff = df.copy()
    dff = pd.DataFrame(get_team_table(option_slctd_team, option_slctd_year))
    print(dff.head())

    colorscale = [[0, 'lightsalmon'], [0.5, 'mediumturquoise'], [1, 'gold']]

    fig = go.Figure(data =
        go.Contour(
            z=dff,
            contours=dict(
            start=0,
            end=8,
            size=1,
            ),
            contours_coloring='heatmap',
            colorscale=colorscale,
            line_smoothing=0.85,
            connectgaps=True, 
        ))  

    return year_container, fig

if __name__=='__main__':
    app.run_server(debug=True)