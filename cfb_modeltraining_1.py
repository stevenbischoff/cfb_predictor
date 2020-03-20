import pandas as pd
import numpy as np
import math
import copy
from cfb_trainfxns import *
from cfb_datafxns import *

def train(first_year, last_year, game_data='adv', window=2,
          train_size=0.8, learn_rate=0.00001, tol=0.002,
          n_learn_rate_changes=2, year_discount=0,
          verbose=True):

  if len(game_data) < 5:
    game_data = data_gather(first_year, last_year+1, game_data)#get rid of +1
  
  game_data, sos = data_init(game_data,first_year,last_year)

  nn_list = []
  for i in range(1,14):
    nn_list.append(NeuralNet((len(game_data.columns)-12)//2,
                         learn_rate, year_discount, tol))   

  i = 1
  while sum([nn.switch for nn in nn_list]) > 0:
    if verbose == True:
      print(i)
    for week in range(1,14):
      nn = nn_list[week-1]
      if nn.switch > 0:
        nn.n_worse += 1
        if week < 13:
          train, test = custom_train_test_split(
            game_data, train_size, first_year, last_year, week-window, week+window)
        else:
          train, test = custom_train_test_split(
            game_data, train_size, first_year, last_year, week-window, week+5)            
        
        nn.epoch(train, last_year)     
        nn.error_check(test, last_year)
        nn.assess(i, 4)
            
        if verbose == True:
          print(week,'Train Error:',round(nn.train_error,5),
                'Test Error:',round(nn.test_error,5))
          
        if week == 13:
          for p in range(2):   
            ratings_calc(sos, nn, game_data)
            sos_calc(sos, game_data, first_year)
    i+=1

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
            train, test = custom_train_test_split(
              game_data, train_size, first_year, last_year, week-window, week+window)
          else:
            train, test = custom_train_test_split(
              game_data, train_size, first_year, last_year, week-window, week+5)            
          
          nn.epoch(train,last_year)     
          nn.error_check(test,last_year)
          nn.assess(j, 3)

          if verbose == True:
            print(week,'Train Error:',round(nn.train_error,5),
                  'Test Error:',round(nn.test_error,5))

          if week == 13:
            for p in range(2):    
              ratings_calc(sos, nn, game_data)
              sos_calc(sos, game_data, first_year)
      i += 1
      j += 1

  return nn_list, sos, game_data
