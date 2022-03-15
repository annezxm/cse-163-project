import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import functools as fc
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
sns.set()


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


# Q1
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


# Q1
def top_5_plot(df):
    '''
    Takes a dataset, plots a line chart of the average attribute rating and
    year, and saves the plot
    '''
    df.reset_index(level=0, inplace=True)
    sns.relplot(x='year', y='rating', hue='attribute', data=df, kind='line')
    plt.xlabel('Year')
    plt.ylabel('Rating')
    plt.title('Key Attribute of Top 5% Players')
    plt.savefig('small_data/top_5_percent_attribute.png', bbox_inches='tight')
    plt.show()


# Q2
def preferred_foot_plot(dataset):
    '''
    Takes a dataset and plots a histogram of preferred foot of players,
    then saves the plot
    '''
    sns.catplot(x='preferred_foot', kind='count', data=dataset)
    plt.title('Preferred Foot of Players')
    plt.xlabel('Foot')
    plt.ylabel('Count')
    plt.savefig('small_data/preferred_foot.png', bbox_inches='tight')
    plt.show()


# Q3
def nationality_cloud(df):
    '''
    Create a Word map according to the number of player's nationality from
    2015-2022. Remove the blank space of some countries for manipulation in
    wordmap function.
    '''
    df.dropna()
    df = df.replace({'nationality_name': {"Korea Republic": "Korea",
                                          "United States": "UnitedStates",
                                          "Republic of Ireland": "Ireland",
                                          "Northern Ireland":
                                          "Northern_Ireland"}})
    text = " ".join(review for review in df.nationality_name.astype(str))
    word_cloud = WordCloud(background_color="white",
                           collocations=False).generate(text)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.savefig("small_data/cloud_map.png")
    plt.show()


# Q3
def nationality_histogram(df):
    '''
    Create a histogram that shows the top 10 countires that
    has most FIFA players in 2022.
    '''
    sns.catplot(x="nationality_name", kind='count', color='b', data=df,
                order=pd.value_counts(df['nationality_name']).iloc[:10].index)
    plt.title('Top 10 countries with most players')
    plt.xticks(rotation=-45)
    plt.xlabel('Nationality')
    plt.ylabel('Count')
    plt.savefig('small_data/most_player_22.png', bbox_inches='tight')
    plt.show()


# Q4 (only height)
def avg_height(df):
    '''
    Group the data by year and then calculate the mean of
    height and weight of each year.
    '''
    df.groupby('year')["height_cm"].mean()
    sns.relplot(x="year", y="height_cm", kind="line", data=df)
    plt.title("Player's average height by year")
    plt.xlabel("Year")
    plt.ylabel("Height(cm)")
    plt.savefig('small_data/avg_height.png')
    plt.show()


# Q4 (height and weight in same figure)
def avg_height_weight(df):
    '''
    Group the data by year. Take only the Big 5 clubs and the starter players
    in the data and then calculate the mean of height and weight of each year.
    Plot them in one figure by two axes
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
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(data.year, data.height_cm, color="red", marker="o")
    ax2.plot(data.year, data.weight_kg, color="blue", marker="o")
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Height(cm)', color="red", fontsize=16)
    ax2.set_ylabel('Weight(kg)', color="blue", fontsize=16)
    plt.title("Player's average height and weight across year")
    plt.savefig('small_data/avg_height_weight.png', bbox_inches='tight')
    plt.show()


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


# Q5
def age_plot(dataset):
    '''
    Takes a dataset and filters to players under 40-years-old, plot
    the distribution as a histogram and saves the plot
    '''
    dataset = dataset[(dataset['age'] <= 40)]
    sns.catplot(x='age', kind='count', color='b', data=dataset)
    plt.title('Age Distribution of Players')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.xticks(rotation=-45)
    plt.savefig('small_data/age_distribution.png', bbox_inches='tight')
    plt.show()


# Q5
def league_wage_plot(dataset):
    '''
    Takes a dataset and selects only players in the Big Five League and age
    between 20 to 40 years old. Plots the distribution of wage by league
    using boxplot and saves the plot
    '''
    mask1 = ((dataset['league_name'] == 'French Ligue 1') |
             (dataset['league_name'] == 'Spain Primera Division') |
             (dataset['league_name'] == 'German 1. Bundesliga') |
             (dataset['league_name'] == 'Italian Serie A') |
             (dataset['league_name'] == 'English Premier League'))
    mask2 = (dataset['age'] <= 40) & (dataset['age'] >= 20)
    dataset = dataset[mask1 & mask2]
    sns.boxplot(x='league_name', y='wage_eur', data=dataset, showfliers=False)
    plt.title('League and Wage Distribution')
    plt.xlabel('League')
    plt.ylabel('Wage')
    plt.xticks(rotation=-45)
    plt.savefig('small_data/league_wage_distribution.png', bbox_inches='tight')
    plt.show()


# Q7
def wage_predict(df, df_22):
    '''
    Takes a list of dataset of players from 2015 to 2021 and a dataset of
    2022. Selects only players from the Big Five League, age under 45, and
    not substitution in the club. Trains a decision tree regressor based
    on datasets from 2015 to 2021, calculates the test mean squared error of
    2015 to 2021 and the mean squared error using the data of 2022.
    Returns the two MSEs in a list
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
    features = data.loc[:, ['overall', 'potential']]
    labels = data['wage_eur']
    features_train, features_test, labels_train, labels_test =\
        train_test_split(features, labels, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(features_train, labels_train)
    test_predictions = model.predict(features_test)
    test_MSE = mean_squared_error(labels_test, test_predictions)
    df_22 = df_22[mask1 & mask2]
    df_22['wage_eur'] = df_22['wage_eur'] / 10000
    df_22 = df_22[["overall", "potential", "value_eur", "wage_eur"]].dropna()
    labels_22 = df_22['wage_eur']
    features_22 = df_22.loc[:, ['overall', 'potential']]
    prediction_22 = model.predict(features_22)
    MSE_22 = mean_squared_error(labels_22, prediction_22)
    return [test_MSE, MSE_22]


def main():
    selected_col = ['short_name', 'overall', 'potential',
                    'value_eur', 'wage_eur', 'age', 'height_cm',
                    'weight_kg', 'league_name', 'club_position',
                    'nationality_name', 'preferred_foot', 'pace',
                    'shooting', 'passing', 'dribbling', 'defending',
                    'physic', 'player_positions']
    data_15 = pd.read_csv('Dataset/players_15.csv')
    data_16 = pd.read_csv('Dataset/players_16.csv')
    data_17 = pd.read_csv('Dataset/players_17.csv')
    data_18 = pd.read_csv('Dataset/players_18.csv')
    data_19 = pd.read_csv('Dataset/players_19.csv')
    data_20 = pd.read_csv('Dataset/players_20.csv')
    data_21 = pd.read_csv('Dataset/players_21.csv')
    data_22 = pd.read_csv('Dataset/players_22.csv')
    data_15 = data_15[selected_col].nlargest(20, 'overall')
    data_16 = data_16[selected_col].nlargest(20, 'overall')
    data_17 = data_17[selected_col].nlargest(20, 'overall')
    data_18 = data_18[selected_col].nlargest(20, 'overall')
    data_19 = data_19[selected_col].nlargest(20, 'overall')
    data_20 = data_20[selected_col].nlargest(20, 'overall')
    data_21 = data_21[selected_col].nlargest(20, 'overall')
    data_22 = data_22[selected_col].nlargest(20, 'overall')
    data_overyear_list = [data_15, data_16, data_17, data_18, data_19,
                          data_20, data_21, data_22]
    merged_data = merge_data(data_overyear_list)
    top_5_plot(top_5(data_overyear_list))
    preferred_foot_plot(data_22)
    age_plot(data_22)
    league_wage_plot(data_22)
    nationality_histogram(data_22)
    nationality_cloud(merged_data)
    avg_height_weight(merged_data)
    avg_height(merged_data)
    data_overyear_list2 = [data_15, data_16, data_17, data_18, data_19,
                           data_20, data_21]
    merged_data2 = merge_data(data_overyear_list2)
    wage_predicted = wage_predict(merged_data2, data_22)
    print("The test MSE for model build on 15 - 22 data is: ",
          wage_predicted[0])
    print("The test MSE using the model 15-21 to predict 2022 data is: ",
          wage_predicted[1])
    print("All test passed")


if __name__ == '__main__':
    main()
