import pandas as pd
import numpy as np
import copy
from cfb_trainfxns import *
from cfb_datafxns import *

def train(first_season, last_season, game_data = 'adv', window = 2, train_size = 0.8, learn_rate = 0.0001, 
          tol = 0.0001, n_learn_rate_changes = 3, season_discount = 0, verbose = True):
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
  if type(game_data) == str:
    game_data = data_gather(first_season, last_season, game_data)

  game_data = data_init(game_data, first_season, last_season)
  ratings_dict = ratings_init(game_data, first_season, last_season)
  
  nn_list = []
  for i in range(1,14):
    nn_list.append(NeuralNet((len(game_data.columns)-12)//2, i, learn_rate, season_discount, tol))   

  i = 1
  counter = 1
  threshold = 6
  for change in range(n_learn_rate_changes + 1):    
    if verbose == True and change > 0:
      print('Learn Rate Change', change)
    while sum([nn.switch for nn in nn_list]) > 0:
      if verbose == True:
        print('Round', i)
        i += 1
      nn_list, game_data, ratings_dict = training_round(nn_list, game_data, ratings_dict, train_size, first_season, last_season, window, counter, threshold, verbose)        
      counter += 1     
    threshold = max(1, threshold - 1)
    counter = 1
    for nn in nn_list:
      nn.switch = 1
      nn.n_worse = 0
      nn.learn *= 0.1
  
  if verbose == True:
    print('\nFinal Week Errors')
    for nn in nn_list:
      print('Train Error:', round(nn.train_error, 5), 'Test Error:', round(nn.best_test_error, 5))

  return nn_list, game_data, ratings_dict


def training_round(nn_list, game_data, ratings_dict, train_size, first_season, last_season, window, counter, threshold, verbose):
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
        print('Week:', week, 'Train Error:', round(nn.train_error,5), 'Test Error:', round(nn.test_error,5))

      if week == 13:
        for p in range(threshold//2):
          ratings_calc(ratings_dict, nn, game_data)        
          sos_calc(ratings_dict, game_data, first_season, last_season)
            
  return nn_list, game_data, ratings_dict


def model_test(nn_list, tot, game_data, ratings, season, verbose = True):
  """
  Assuming the model was trained on data up to final_season, this function tests the model on final_season + 1
  """
  game_data_season = tot.loc[(tot.season == season),:].copy()

  ratings_season = ratings_init(game_data_season, season, season)
  
  for team in ratings.keys():
    rating = ratings[team][str(season-1) + 'Rating']
    game_data_season.loc[game_data_season.home_team == team, 'home_last_rating'] = rating
    game_data_season.loc[game_data_season.away_team == team, 'away_last_rating'] = rating

  for p in range(4):
    ratings_calc(ratings_season, nn_list[-1], game_data_season)
    sos_calc(ratings_season, game_data_season, season, season)
    print(pd.DataFrame.from_dict(ratings_season, orient = 'index').sort_values('2019Rating',ascending=False).head(30))

  for i in range(1, 14):
    nn = nn_list[i-1]
    if i == 1:
      game_data_temp = game_data_season[game_data_season.week <= 1]
    elif i == 13:
      game_data_temp = game_data_season[(game_data_season.week >= 13)&(game_data_season.week < 20)]
    else:
      game_data_temp = game_data_season[game_data_season.week == i]
    nn.error_check(game_data_temp, season)
    if verbose == True:
      print('Week:', i, 'Error:', nn.test_loss/nn.count)
  if verbose == True:
    print('Total Error:',sum([nn.test_loss for nn in nn_list])/sum([nn.count for nn in nn_list]))

  return game_data_season, ratings_season, nn_list
