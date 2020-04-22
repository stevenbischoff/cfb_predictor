import pandas as pd
import numpy as np
import copy
import cfb_predictor.train_fxns as train_fxns
import cfb_predictor.data_fxns as data_fxns
import cfb_predictor.neural_network as neural_network
import cfb_predictor.cfg as cfg
import cfb_predictor.model_testing as model_testing

def train(first_season, last_season, game_data = 'adv', window = 2, train_size = 0.8, learn_rate = 0.00005, 
          tol = 0.001, n_learn_rate_changes = 2, season_discount = 0.0, week_discount = 0.0, pca = False,
          verbose = False):
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
    week_discount: float
      If non-zero, statistics from recent weeks are weighted higher than statistics from less recent weeks.
    verbose: bool
  """  
  cfg.init(week_discount, first_season, last_season)
  
  if type(game_data) == str:
    game_data = data_fxns.data_gather(game_data, verbose)

  game_data = data_fxns.data_init(game_data, pca)
  cfg.n_cols = (len(game_data.columns)-13)//2
  data_fxns.ratings_init(game_data)
  data_fxns.index_dict_init(game_data)
  data_fxns.nn_list_init(learn_rate, season_discount, tol, window)

  for nn in cfg.nn_list:
    nn.init_train_test(game_data, train_size)
    if verbose == True:
      print('Week:', nn.week)
    for change in range(n_learn_rate_changes + 1):
      if verbose == True and change > 0:
        print('Learn Rate Change', change)
      while nn.switch > 0:
        game_data = training_round(game_data, train_size, nn, verbose)
        cfg.learn_rate_counter += 1
      cfg.threshold = max(1, cfg.threshold - 1)
      cfg.learn_rate_counter = 1

      nn.reset()
      
    cfg.threshold = 6
    
  if verbose == True:
    print('\nFinal Week Errors')
    for nn in cfg.nn_list:
      print(nn)
    season_error = sum([nn.total_test_error for nn in cfg.nn_list])/sum([nn.count for nn in cfg.nn_list])
    print('Total Test Error:', season_error)

  return game_data
   

def training_round(game_data, train_size, nn, verbose):
  nn.n_worse += 1
  nn.epoch(nn.train)     
  nn.error_check(nn.test)
  nn.assess(cfg.learn_rate_counter, cfg.threshold)

  if verbose == True:
    print(nn)

  if nn.week == 13:
    train_fxns.ratings_sos_calculation(nn, game_data, cfg.threshold//2)        
            
  return game_data
