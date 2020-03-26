import pandas as pd
import numpy as np
import copy
from cfb_trainfxns import *
from cfb_datafxns import *


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
  global learn_rate_counter, threshold, index_dict
  learn_rate_counter, threshold, index_dict = 1, 6, {}  
  
  if type(game_data) == str:
    game_data = data_gather(first_season, last_season, game_data)

  game_data = data_init(game_data, first_season, last_season)
  ratings_dict = ratings_init(game_data, first_season, last_season)
  
  for team in set(list(game_data.home_team) + list(game_data.away_team)):
    index_dict[team] = {}
    for season in range(first_season, last_season+1):
      home_indices = game_data[(game_data.season == season)&(game_data.home_team == team)].index
      away_indices = game_data[(game_data.season == season)&(game_data.away_team == team)].index
      index_dict[team][season] = {'home': home_indices, 'away': away_indices}
      
  
  nn_list = []
  for i in range(1,14):
    nn_list.append(NeuralNet((len(game_data.columns)-12)//2, i, learn_rate, season_discount, tol))   

  total_rounds = 1
  for change in range(n_learn_rate_changes + 1):    
    if verbose == True and change > 0:
      print('Learn Rate Change', change)
    while sum([nn.switch for nn in nn_list]) > 0:
      if verbose == True:
        print('Round', total_rounds)
        total_rounds += 1
      nn_list, game_data, ratings_dict = training_round(nn_list, game_data, ratings_dict, train_size, first_season, last_season, window, verbose)        
      learn_rate_counter += 1     
    threshold = max(1, threshold - 1)
    counter = 1
    for nn in nn_list:
      nn.switch = 1
      nn.n_worse = 0
      nn.learn *= 0.1
  
  if verbose == True:
    print('\nFinal Week Errors')
    for nn in nn_list:
      print(nn)

  return nn_list, game_data, ratings_dict


def training_round(nn_list, game_data, ratings_dict, train_size, first_season, last_season, window, verbose):
  global learn_rate_counter, threshold
  print(learn_rate_counter)
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
      nn.assess()

      if verbose == True:
        print(nn)

      if week == 13:
        ratings_sos_calculation(nn, ratings_dict, game_data, first_season, last_season, threshold//2)        
            
  return nn_list, game_data, ratings_dict
  

def model_test(nn_list, tot, game_data, ratings, season, verbose = True):
  """
  Assuming the model was trained on data up to final_season, this function tests the model on final_season + 1.
  """
  game_data_season = tot.loc[(tot.season == season),:].copy()
  ratings_season = ratings_init(game_data_season, season, season)
  
  index_dict = {}
  for team in set(list(game_data_season.home_team) + list(game_data.away_team)):
    index_dict[team] = {}
    for season in range(season, season+1):
      home_indices = game_data_season[(game_data_season.season == season)&(game_data_season.home_team == team)].index
      away_indices = game_data_season[(game_data_season.season == season)&(game_data_season.away_team == team)].index
      index_dict[team][season] = {'home': home_indices, 'away': away_indices}
  
  for team in ratings.keys():
    rating = ratings[team][str(season-1) + 'Rating']
    game_data_season.loc[game_data_season.home_team == team, 'home_last_rating'] = rating
    game_data_season.loc[game_data_season.away_team == team, 'away_last_rating'] = rating

  ratings_sos_calculation(nn_list[-1], ratings_season, game_data_season, season, season, 4)

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


def ratings_sos_calculation(nn, ratings_dict, game_data, first_season, last_season, rounds):
  """
  Performs a specified number of ratings and SOS updates.
  """
  for r in range(rounds):
    ratings_calculation(nn, ratings_dict, game_data)        
    sos_calculation(ratings_dict, game_data, first_season, last_season)


tot = pd.read_pickle(
  r'C:\Users\Visitor\AppData\Local\Programs\Python\Python38-32\cfbdata\cfb619_seasondata.pkl')

nn_list, game_data, ratings = train(2013, 2018, game_data = tot, learn_rate = 0.00008, season_discount = 0)
gds, ratingss, nn_list = model_test(nn_list, tot, game_data, ratings, 2019)
