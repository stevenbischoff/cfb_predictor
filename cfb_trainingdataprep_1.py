import pandas as pd

def data_init(game_data,first_year,last_year):

  game_data = game_data[(game_data.season >= first_year)&
                        (game_data.season <= last_year+1)] # get rid of +1
  
  game_data.insert((len(game_data.columns)-12)//2 + 12, 'home_last_rating', 0.5)
  game_data.insert((len(game_data.columns)-12)//2 + 12, 'home_SOS', 0.45)
  game_data.insert(len(game_data.columns), 'away_last_rating', 0.5)
  game_data.insert(len(game_data.columns), 'away_SOS', 0.45)

  teams = list(set(list(game_data.home_team)+list(game_data.away_team)))

  sos = pd.DataFrame(index=[i for i in range(len(teams))],
                     columns=['Team']+[
                       str(season)+'SOS' for season in range(first_year, last_year+2)]+[#+1
                         str(season)+'Rating' for season in range(first_year, last_year+2)],
                     dtype = 'float32')

  sos['Team'] = teams
  for season in range(first_year, last_year+2):#+1
    game_data_season = game_data[game_data.season == season]
    sos_list = []
    rating_list = []
    for team in teams:
      if len(game_data_season[game_data_season.home_team==team])>2 or len(
        game_data_season[game_data_season.away_team==team])>3:
        sos_list.append(0.45)
        rating_list.append(0.5)
      else:
        sos_list.append('FCS')
        rating_list.append(0.15)
    sos[str(season)+'SOS'] = sos_list
    sos[str(season)+'Rating'] = rating_list
    
  game_data = game_data.sample(frac=1).reset_index(drop=True)

  return game_data, sos

def custom_train_test_split(game_data, train_size, first_year, last_year,
                            first_week, last_week):
  game_data_range = game_data[
    ((game_data.week>=first_week)&(game_data.week<=min(19,last_week)))&
    (game_data.season>=first_year)&(game_data.season<=last_year)
    ].reset_index(drop=True)

  train = game_data_range.iloc[:int(len(game_data_range)*train_size)
    ].sample(frac=1).reset_index(drop=True) 
  test = game_data_range.iloc[int(len(game_data_range)*train_size):
    ].reset_index(drop=True)

  return train, test
  
  

  
