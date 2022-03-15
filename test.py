'''
Name: Yuchen Wu, Xinmeng Zhang
This is a test file using a smaller data size to check the accuracy
of the output generated in the Project_main program.
'''
import pandas as pd
import seaborn as sns
import Project_main as pj1
import Project_data_processing as pj2
sns.set()


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
    merged_data = pj2.merge_data(data_overyear_list)
    pj1.top_5_plot(pj2.top_5(data_overyear_list), 'small_data/')
    pj1.preferred_foot_plot(data_22, 'small_data/')
    pj1.age_plot(data_22, 'small_data/')
    pj1.league_wage_plot(data_22, 'small_data/')
    pj1.nationality_histogram(data_22, 'most_player_22.png',
                              'Top 10 countries with most players',
                              'small_data/')
    pj1.nationality_cloud(merged_data, 'small_data/')
    pj1.avg_height_weight(merged_data, 'small_data/')
    data_overyear_list2 = [data_15, data_16, data_17, data_18, data_19,
                           data_20, data_21]
    merged_data2 = pj2.merge_data(data_overyear_list2)
    wage_predicted = pj1.wage_predict(merged_data2, data_22)
    print("The test MSE for model build on 15 - 22 data is: ",
          wage_predicted[0])
    print("The test MSE using the model 15-21 to predict 2022 data is: ",
          wage_predicted[1])


if __name__ == '__main__':
    main()
