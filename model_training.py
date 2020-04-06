import pandas as pd
import numpy as np
import copy
import train_fxns
import data_fxns
import cfg


def train(first_season, last_season, game_data = 'adv', window = 2, train_size = 0.8, learn_rate = 0.0001, 
          tol = 0.001, n_learn_rate_changes = 3, season_discount = 0, verbose = True):
  """
  The full training algorithm
  ----------
  Parameters
  ----------
    first_season, last_season: int
    game_data: str or DataFrame
      If str, the data_gather function is called to scrape and prepare the correct type of data
    window: int
      The NeuralNet for week i is trained on game_data from weeks in the range [i - window, i + window]
    train_size: float
      The proportion of the data to be used for training
    learn_rate: float
    tol: float
      The tolerance for early stopping
    n_learn_rate_changes: int
      The number of times that learn_rate changes when the early stopping conditions are met
    season_discount: float
      If non-zero, a data point's contribution to total loss is discounted based on the season it came from. The loss
      is multiplied by 1 - season_discount*(final_season - data_point_season).
    verbose: bool
  """
  cfg.init()
  
  if type(game_data) == str:
    game_data = data_fxns.data_gather(first_season, last_season, game_data)

  n_cols = (len(game_data.columns)-12)//2

  game_data = data_fxns.data_init(game_data, first_season, last_season)
  data_fxns.ratings_init(game_data, first_season, last_season)
  data_fxns.index_dict_init(game_data, first_season, last_season)
  data_fxns.nn_list_init(n_cols, learn_rate, season_discount, tol, window)

  for change in range(n_learn_rate_changes + 1):
    if verbose == True and change > 0:
      print('Learn Rate Change', change)
    while sum([nn.switch for nn in cfg.nn_list]) > 0:
      if verbose == True:
        print('Round', total_rounds)
        total_rounds += 1
      game_data = training_round(game_data, train_size, first_season, last_season, verbose)
      cfg.learn_rate_counter += 1
    cfg.threshold = max(1, cfg.threshold - 1)
    cfg.learn_rate_counter = 1
    for nn in cfg.nn_list:
      nn.switch = 1
      nn.n_worse = 0
      nn.learn *= 0.2
  
  if verbose == True:
    print('\nFinal Week Errors')
    for nn in nn_list:
      print(nn)

  return game_data


def training_round(game_data, train_size, first_season, last_season, verbose):
  for nn in cfg.nn_list:
    if nn.switch > 0:
      nn.n_worse += 1
      if nn.week < 13:
        train, test = data_fxns.custom_train_test_split(game_data, train_size, nn.week-nn.window, nn.week+nn.window)
      else:
        train, test = data_fxns.custom_train_test_split(game_data, train_size, nn.week-nn.window, nn.week+6)            

      nn.epoch(train, last_season)     
      nn.error_check(test, last_season)
      nn.assess(cfg.learn_rate_counter, cfg.threshold)

      if verbose == True:
        print(nn)

      if nn.week == 13:
        train_fxns.ratings_sos_calculation(nn, game_data, first_season, last_season, cfg.threshold//2)        
            
  return game_data
  

def model_test(tot, game_data, season, verbose = True):
  """
  Assuming the model was trained on data up to final_season, this function tests the model on final_season + 1.
  """
  #global ratings_dict
  game_data_season = tot.loc[(tot.season == season),:].copy()
  data_fxns.ratings_init(tot, season, season)
  data_fxns.index_dict_init(game_data_season, season, season)
    
  for team in cfg.ratings_dict.keys():
    try:
      rating = cfg.ratings_dict[team][str(season-1) + 'Rating']
      game_data_season.loc[game_data_season.home_team == team, 'home_last_rating'] = rating
      game_data_season.loc[game_data_season.away_team == team, 'away_last_rating'] = rating
    except:
      pass

  train_fxns.ratings_sos_calculation(cfg.nn_list[-1], game_data_season, season, season, 4)

  for nn in cfg.nn_list:
    if nn.week == 1:
      game_data_temp = game_data_season[game_data_season.week <= 1].reset_index(drop=True)
    elif nn.week == 13:
      game_data_temp = game_data_season[(game_data_season.week >= 13)&(game_data_season.week < 20)].reset_index(drop=True)
    else:
      game_data_temp = game_data_season[game_data_season.week == nn.week].reset_index(drop=True)
    results = nn.error_check(game_data_temp, season)
    if verbose == True:
      print('Week', nn.week, 'Error:', nn.test_loss/nn.count, nn.count)
  season_error = sum([nn.test_loss for nn in cfg.nn_list])/sum([nn.count for nn in cfg.nn_list])
  if verbose == True:
    print('Total Error:', season_error)

  return game_data_season, season_error
