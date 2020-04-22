import pandas as pd
import numpy as np
import copy
import cfg
import data_fxns
import train_fxns


def model_test(test_season, game_data_season = 'adv', verbose = True):
  """
  This function is intended to be used after the model is trained on the seasons up to test_season.
  It plugs in last_ratings and performs SOS calculations before calculating the model's mean absolute
  error in test_season.
  """
  #####
  vegas = pd.read_pickle(
    r'C:\Users\Visitor\AppData\Local\Programs\Python\Python38-32\cfbdata\cfb_betting2019_clean.pkl')
  vegas['my_pred'] = np.nan
  vegas['my_error'] = np.nan
  #####
  cfg.first_season = test_season
  cfg.last_season = test_season

  if type(game_data_season) == str:
    game_data_season = data_fxns.data_gather(game_data_season)

  game_data_season = game_data_season[game_data_season.season == test_season].copy()  
  data_fxns.ratings_init(game_data_season)
  data_fxns.index_dict_init(game_data_season)
    
  for team in cfg.ratings_dict.keys():
    try:
      rating = cfg.ratings_dict[team][str(test_season-1) + 'Rating']
      game_data_season.loc[game_data_season.home_team == team, 'home_last_rating'] = rating
      game_data_season.loc[game_data_season.away_team == team, 'away_last_rating'] = rating
    except:
      pass
  
  train_fxns.ratings_sos_calculation(cfg.nn_list[0], game_data_season, 6)

  total_loss = 0#
  for nn in cfg.nn_list:
    if nn.week < 13:
      game_data_week = game_data_season[game_data_season.week == nn.week].reset_index(drop=True)
    else:
      game_data_week = game_data_season[(game_data_season.week > 12)&(game_data_season.week < 20)].reset_index(drop=True)
      
    nn.error_check(game_data_week)
    #####
    for i in range(len(game_data_week)):
      game = game_data_week.iloc[i]
      if game.away_conference == None:
        continue
      elif game.home_conference == None:
        continue
      
      r1 = nn.feedforward(game[13:cfg.n_cols + 13].astype('float32'))
      r2 = nn.feedforward(game[cfg.n_cols + 13:].astype('float32'))

      neutral = game.neutral      
      y_pred = nn.margin_predict(r1, r2, neutral)

      vegas.loc[(vegas.id==game.id), 'my_pred'] = y_pred
      vegas.loc[(vegas.id==game.id), 'my_error'] = game.y_actual - y_pred
      total_loss += game.y_actual - y_pred
    #####
    if verbose == True:
      print('Week', nn.week, 'Error:', nn.total_test_error/nn.count, nn.count)
  season_error = sum([nn.total_test_error for nn in cfg.nn_list])/sum([nn.count for nn in cfg.nn_list])
  print(total_loss/sum([nn.count for nn in cfg.nn_list]))#
  print('Total Error:', season_error, sum([nn.count for nn in cfg.nn_list]))

  return game_data_season, season_error, vegas
