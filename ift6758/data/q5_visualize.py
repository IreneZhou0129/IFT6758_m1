# import json
# import pprint
import pandas as pd
import numpy as np
import csv
import os
import glob
import chardet
import matplotlib.pyplot as plt
import math

extension = 'csv'
years = [2016, 2017, 2018, 2019, 2020]
# years = [2016]
csv_data_path = r"D:\Python\IFT6758\CSV_data"
visualize_path = r"D:\Python\IFT6758\Visualize"

# Check the encoding type
# with open(r'D:\Python\IFT6758\CSV_data\playoffs\2016\2016030111.csv', 'rb') as f:
#     result = chardet.detect(f.read()) # or readline if the file is large
#     print(result['encoding'])
       # return ISO-8859-1

def combine_csv():
    for year in years:
        # Combine all .csv files for playoffs
        # set working directory for playoffs + year
        os.chdir(fr"{csv_data_path}\playoffs\{year}")

        # use glob to match the pattern 'csv'
        all_files = [i for i in glob.glob('*.{}'.format(extension))]

        # combine all files in the list
        # set encoding to 'ISO-8859-1' per the previous catching return
        combined_csv = pd.concat([pd.read_csv(f, encoding='ISO-8859-1') for f in all_files])

        os.chdir(visualize_path)
        combined_csv.to_csv(f"{year}_playoffs.csv", index=False, encoding='ISO-8859-1')

        # Combine all .csv files for regular_season
        # set working directory for regular_season + year
        os.chdir(fr"{csv_data_path}\regular_season\{year}")

        # use glob to match the pattern 'csv'
        all_files = [i for i in glob.glob('*.{}'.format(extension))]

        # combine all files in the list
        # set the encoding to 'ISO-8859-1' per the previous catching return
        combined_csv = pd.concat([pd.read_csv(f, encoding='ISO-8859-1') for f in all_files])

        os.chdir(visualize_path)
        combined_csv.to_csv(f"{year}_regular_season.csv", index=False, encoding='ISO-8859-1')

        # Combine two intermediate .csv files
        combine_files = [f"{year}_playoffs.csv", f"{year}_regular_season.csv"]
        combined_csv = pd.concat([pd.read_csv(f, encoding='ISO-8859-1') for f in combine_files])
        combined_csv.to_csv(f"{year}.csv", index=False, encoding='ISO-8859-1')

# combine_csv()

def goal_vs_shot_type():
    # Select one year (2018) to analyze
    df = pd.read_csv(r'D:\Python\IFT6758\Visualize\2018.csv', encoding='ISO-8859-1')
    # df.head(10)

    # Group by 'Shot Type' and 'Shot or Goal', add a 'Count' for the sum up
    df['Count'] = 1
    group_data = df.groupby(['Shot Type', 'Shot or Goal'])['Count'].sum().reset_index(name="Count")
    print(group_data)

    # Generate the figure for Q5.1
    # group_data.pivot("Shot Type","Shot or Goal","Count")[["Shot", "Goal"]].plot.bar(stacked=True, color=["blue", "red"], legend=False)
    group_data.pivot("Shot Type","Shot or Goal","Count")[["Shot", "Goal"]].plot.bar(stacked=True, color=["red", "blue"], legend=False, width=1)

# analyze the relationship between goal and distance
def goal_vs_distance(year):
    # Select three years (2018, 2019, 2020) to analyze
    # year 2018
    df = pd.read_csv(fr'D:\Python\IFT6758\Visualize\{year}.csv', encoding='ISO-8859-1')

    # Simplify column name
    df = df.rename(columns={'X-Coordinate': 'X', 'Y-Coordinate': 'Y', 'Home or Away': 'Home'})

    # Calculate Distance
    # df['Distance'] = np.where((df.Home=='Home') ^ (df.Period%2==1), np.sqrt((df.X + 89.) ** 2 + df.Y ** 2), np.sqrt((df.X - 89.) ** 2 + df.Y ** 2))
    # df['Distance'] = np.where(df.X < 0., np.sqrt((df.X + 89.) ** 2 + df.Y ** 2), np.sqrt((df.X - 89.) ** 2 + df.Y ** 2))
    """
    X_Left_Net: -89
    Y_Left_Net: 0
    X_Right_Net: 89
    Y_Right_Net: 0
    """
    home, p_odd = df.Home == 'Home', df.Period % 2 ==1
    condition=[home&p_odd, home&~p_odd, ~home&p_odd]
    # if Home and odd period, Net is left
    choices = [np.sqrt((df.X+89.)**2 + df.Y**2), np.sqrt((df.X-89.)**2 + df.Y**2), np.sqrt((df.X+89.)**2 + df.Y**2)]
    df['Distance']=np.select(condition, choices, np.sqrt((df.X-89.)**2 + df.Y**2))
    print(df.head(2))

    # Generate the histogram of chance of goal vs distance
    df2 = df.pivot_table(index='Distance', columns='Shot or Goal', aggfunc='size', fill_value=0)
    bin_range = range(0, int(df2.index.max())+1, 10)
    df3 = df2.groupby(pd.cut(df2.index, bins=bin_range)).sum()
    df3['Goal'].div(df3.sum(1)).plot.bar(width=1)

goal_vs_distance(2018)


