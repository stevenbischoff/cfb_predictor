import pandas as pd
import numpy as np
import copy
import cfb_predictor.cfg as cfg
import cfb_predictor.data_fxns as data_fxns
import cfb_predictor.train_fxns as train_fxns


def model_test(test_season, game_data_season = 'adv', verbose = False):
  """
  This function is intended to be used after the model is trained on the seasons up to test_season.
  It plugs in last_ratings and performs SOS calculations before calculating the model's mean absolute
  error in test_season.
  """
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

  for nn in cfg.nn_list:
    if nn.week < 13:
      game_data_week = game_data_season[game_data_season.week == nn.week].reset_index(drop=True)
    else:
      game_data_week = game_data_season[(game_data_season.week > 12)&(game_data_season.week < 20)].reset_index(drop=True)
      
    nn.error_check(game_data_week)
    
    if verbose == True:
      print('Week', nn.week, 'Error:', nn.total_test_error/nn.count, nn.count)
  season_error = sum([nn.total_test_error for nn in cfg.nn_list])/sum([nn.count for nn in cfg.nn_list])
  print('Total Error:', season_error, sum([nn.count for nn in cfg.nn_list]))

  return game_data_season, season_error
