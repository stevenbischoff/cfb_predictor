import pandas as pd
import numpy as np
import copy
from cfb_trainfxns import *
from cfb_datafxns import *

def train(first_season, last_season, game_data='adv', window=2, train_size=0.8, learn_rate=0.0001, 
          tol=0.0001, n_learn_rate_changes=3, season_discount=0, verbose=True):
  """
  The full training algorithm
  -----
  Notes
  -----
  
  ----------
  Parameters
  ----------
    first_season, last_season: int
    game_data: str or DataFrame
      If str, the data_gather function is called to scrape and prepare the data
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

  if type(game_data) == str:
    game_data = data_gather(first_season, last_season, game_data)
  
  sos = sos_init(game_data, first_season, last_season)

  nn_list = []
  for i in range(1,14):
    nn_list.append(NeuralNet((len(game_data.columns)-12)//2, learn_rate, season_discount, tol))   

  i = 1
  counter = 1
  threshold = 6
  for change in range(n_learn_rate_changes + 1):
    while sum([nn.switch for nn in nn_list]) > 0:
      if verbose == True:
        print(i)
      nn_list, game_data, sos = training_round(nn_list, game_data, sos, window, counter, threshold, verbose)
      counter += 1
      i += 1
    threshold = max(1, threshold - 1)
    for nn in nn_list:
      nn.switch = 1
      nn.n_worse = 0
      nn.learn *= 0.1
      
    """for week in range(1,14):
      nn = nn_list[week-1]
      if nn.switch > 0:
        nn.n_worse += 1
        if week < 13:
          train, test = custom_train_test_split(game_data, train_size, week-window, week+window)
        else:
          train, test = custom_train_test_split(game_data, train_size, week-window, week+6)            
        
        nn.epoch(train, last_season)     
        nn.error_check(test, last_season)
        nn.assess(i, 8)
            
        if verbose == True:
          print(week, 'Train Error:', round(nn.train_error,5), 'Test Error:', round(nn.test_error,5))
          
        if week == 13:
          for p in range(2):   
            ratings_calc(sos, nn, game_data, last_season)
            sos_calc(sos, game_data, first_season)
    
    i += 1

  for change in range(n_learn_rate_changes):
    if verbose == True:
      print('Learn Rate Change',change+1)

    for nn in nn_list:
      nn.switch = 1
      nn.n_worse = 0
      nn.learn *= 0.1

    j=1
    while sum([nn.switch for nn in nn_list]) > 0:
      if verbose == True:
        print(i)
      for week in range(1,14):
        nn = nn_list[week-1]
        if nn.switch > 0:
          nn.n_worse += 1
          if week < 13:
            train, test = custom_train_test_split(game_data, train_size, week-window, week+window)
          else:
            train, test = custom_train_test_split(game_data, train_size, week-window, week+6)            
          
          nn.epoch(train,last_season)     
          nn.error_check(test,last_season)
          nn.assess(j, 3)

          if verbose == True:
            print(week, 'Train Error:', round(nn.train_error,5), 'Test Error:', round(nn.test_error,5))

          if week == 13:
            for p in range(2):    
              ratings_calc(sos, nn, game_data, last_season)
              sos_calc(sos, game_data, first_season)
      i += 1
      j += 1"""
  
  if verbose == True:
    print()
    print('Final Week Errors')
    for nn in nn_list:
      print('Train Error:', round(nn.train_error,5), 'Test Error:', round(nn.test_error,5))

  return nn_list, sos, game_data

def training_round(nn_list, game_data, sos, window, counter, threshold, verbose):
  for week in range(1,14):
    nn = nn_list[week-1]
    if nn.switch > 0:
      nn.n_worse += 1
      if week < 13:
        train, test = custom_train_test_split(game_data, train_size, week-window, week+window)
      else:
        train, test = custom_train_test_split(game_data, train_size, week-window, week+6)            

      nn.epoch(train, last_season)     
      nn.error_check(test, last_season)
      nn.assess(counter, threshold)

      if verbose == True:
        print(week, 'Train Error:', round(nn.train_error,5), 'Test Error:', round(nn.test_error,5))

      if week == 13:
        for p in range(2):   
          ratings_calc(sos, nn, game_data, last_season)
          sos_calc(sos, game_data, first_season)
            
  return nn_list, game_data, sos
