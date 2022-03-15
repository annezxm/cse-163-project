"""
Name: Yuchen Wu, Xinmeng Zhang
Does most of the data processing such as merging and selecting
top 5% data which can be used directly in Project_main
"""
import pandas as pd
import functools as fc


# Part1
def merge_data(datalist):
    '''
    Take the 2015-2022 datalist and merge them together. Return a dataframe.
    '''
    year = 2015
    for df in datalist:
        df["year"] = year
        year += 1
    merged = fc.reduce(lambda left, right: pd.merge(left, right, how='outer'),
                       datalist)
    return merged


# Q1 Part1
def top_5_percent_data(data, year):
    '''
    Read a dataset, get the average value of pace, shooting, passing,
    dribbling, defending, and physic of the top 5 percent players by
    overall rating. Take an integer as the year of the dataset and
    returns a dataframe of attribute rating and year
    '''
    top = data.nlargest(int(0.05 * len(data)), 'overall')
    top = top[(top['player_positions'] != 'GK')]
    top = top[['pace', 'shooting', 'passing', 'dribbling',
               'defending', 'physic']]
    average_list = [['pace', top['pace'].mean(), year],
                    ['shooting', top['shooting'].mean(), year],
                    ['passing', top['passing'].mean(), year],
                    ['dribbling', top['dribbling'].mean(), year],
                    ['defending', top['defending'].mean(), year],
                    ['physic', top['physic'].mean(), year]]
    average = pd.DataFrame(average_list, columns=['attribute', 'rating',
                                                  'year'])
    return average


# Q1 Part1
def top_5(data_list):
    '''
    Take a list of datasets and returns a merged dataset of average attribute
    rating and year of top 5 percent players
    '''
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
    top_5_list = []
    for df, year in zip(data_list, years):
        top_5_list.append(top_5_percent_data(df, year))
    data = top_5_list[0]
    top_5_list = top_5_list[1:]
    for df in top_5_list:
        data = data.append(df)
    return data


# Q4 Part1
def avg_heighht_weight_data(df):
    '''
    Takes a dataset, select players from the big five league
    and not a substitution in the club. group the data and
    calculates the average height and weight of players each
    year. Returns the new dataset.
    '''
    big5 = ((df['league_name'] == 'French Ligue 1') |
            (df['league_name'] == 'Spain Primera Division') |
            (df['league_name'] == 'German 1. Bundesliga') |
            (df['league_name'] == 'Italian Serie A') |
            (df['league_name'] == 'English Premier League'))
    starter = df["club_position"] != "SUB"
    data = df[big5 & starter]
    columns = ["height_cm", "weight_kg"]
    data = data.groupby('year', as_index=False)[columns].mean()
    return data


# Q5 Part1
def league_wage_data(df):
    '''
    Takes a dataset, selects players in the big five league
    and age between 20 and 40 inclusively. Returns the new
    dataset
    '''
    mask1 = ((df['league_name'] == 'French Ligue 1') |
             (df['league_name'] == 'Spain Primera Division') |
             (df['league_name'] == 'German 1. Bundesliga') |
             (df['league_name'] == 'Italian Serie A') |
             (df['league_name'] == 'English Premier League'))
    mask2 = (df['age'] <= 40) & (df['age'] >= 20)
    dataset = df[mask1 & mask2]
    return dataset


# Q7 Part1
def wage_predict_data(df):
    '''
    Takes a dataset, selects players in the big five league, age
    under 45, and not substitution in the club. Divides the wage_eur
    of the dataset by 10000. Returns the new dataset
    '''
    mask1 = ((df['league_name'] == 'French Ligue 1') |
             (df['league_name'] == 'Spain Primera Division') |
             (df['league_name'] == 'German 1. Bundesliga') |
             (df['league_name'] == 'Italian Serie A') |
             (df['league_name'] == 'English Premier League'))
    mask2 = (df['age'] <= 45) & (df['club_position'] != 'SUB')
    data = df[mask1 & mask2]
    data = data[["overall", "potential", "value_eur", "wage_eur"]].dropna()
    data['wage_eur'] = data['wage_eur'] / 10000
    return data
