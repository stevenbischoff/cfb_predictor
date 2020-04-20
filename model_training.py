import pandas as pd
import numpy as np
import copy
#import train_fxns
#import data_fxns
#import neural_network
#import cfg

def train(first_season, last_season, game_data = 'adv', window = 2, train_size = 0.8, learn_rate = 0.0001, 
          tol = 0.001, n_learn_rate_changes = 2, season_discount = 0.0, week_discount = 0.0, pca = False,
          verbose = True):
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
    game_data = data_fxns.data_gather(game_data)

  game_data = data_fxns.data_init(game_data, pca)
  cfg.n_cols = (len(game_data.columns)-13)//2
  data_fxns.ratings_init(game_data)
  data_fxns.index_dict_init(game_data)
  data_fxns.nn_list_init(learn_rate, season_discount, tol, window)

  for n in range(len(cfg.nn_list),0,-1):
    nn = cfg.nn_list[n-1]
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
    
  if verbose == False:
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
  

def model_test(tot, game_data, test_season, verbose = True):
  """
  This function is intended to be used after the model is trained on the seasons up to test_season.
  It plugs in last_ratings and performs SOS calculations before calculating the model's absolute error
  in test_season.
  """
  #####
  vegas = pd.read_pickle(
    r'C:\Users\Visitor\AppData\Local\Programs\Python\Python38-32\cfbdata\cfb_betting2019_clean.pkl')
  vegas['my_pred'] = np.nan
  vegas['my_error'] = np.nan
  #####
  cfg.first_season = test_season
  cfg.last_season = test_season
  
  game_data_season = tot[(tot.season == test_season)].copy().reset_index(drop=True)
  data_fxns.ratings_init(tot)
  data_fxns.index_dict_init(game_data_season)
    
  for team in cfg.ratings_dict.keys():
    try:
      rating = cfg.ratings_dict[team][str(test_season-1) + 'Rating']
      game_data_season.loc[game_data_season.home_team == team, 'home_last_rating'] = rating
      game_data_season.loc[game_data_season.away_team == team, 'away_last_rating'] = rating
    except:
      pass
  
  train_fxns.ratings_sos_calculation(cfg.nn_list[-1], game_data_season, 5)

  tot_loss = 0
  for nn in cfg.nn_list:
    if nn.week < 13:
      game_data_week = game_data_season[game_data_season.week == nn.week].reset_index(drop=True)
    else:
      game_data_week = game_data_season[(game_data_season.week > 12)&(game_data_season.week < 20)].reset_index(drop=True)
      
    results = nn.error_check(game_data_week)
    #####
    for i in range(len(game_data_week)):
      game = game_data_week.iloc[i]
      if game.away_conference == None:
        continue
      elif game.home_conference == None:
        continue
      
      X1 = game[13:cfg.n_cols + 13].astype('float32')
      X2 = game[cfg.n_cols + 13:].astype('float32')
      
      r1 = nn.feedforward(X1)
      r2 = nn.feedforward(X2)

      neutral = game.neutral      
      y_pred = nn.margin_predict(r1, r2, neutral)

      vegas.loc[(vegas.id==game.id), 'my_pred'] = y_pred
      vegas.loc[(vegas.id==game.id), 'my_error'] = game.y_actual - y_pred
      tot_loss += game.y_actual - y_pred
    #####
    if verbose == True:
      print('Week', nn.week, 'Error:', nn.total_test_error/nn.count, nn.count)
  season_error = sum([nn.total_test_error for nn in cfg.nn_list])/sum([nn.count for nn in cfg.nn_list])
  if verbose == True:
    print(tot_loss/sum([nn.count for nn in cfg.nn_list]),tot_loss)
    print('Total Error:', season_error, sum([nn.count for nn in cfg.nn_list]))

  return game_data_season, season_error, vegas

"""tot_error = 0
n = 6
fy = 2008
import sys
from scipy.stats import norm
import math
for i in range(n):
  print(i+1)

  tot = pd.read_pickle(
    r'C:\Users\Visitor\AppData\Local\Programs\Python\Python38-32\cfbdata\cfb619_advseasondata.pkl')

  tot = data_fxns.pca(tot, 0.99)
  
  game_data = train(fy,2018,game_data=tot,learn_rate=0.00003,season_discount=0.0,window=2,
                    week_discount=0.0, verbose=False)
                    #n_learn_rate_changes = 0,tol=0.1)
  gds, season_error, vegas = model_test(tot, game_data, 2019)
  tot_error += season_error

  ratings_df = pd.DataFrame.from_dict(cfg.ratings_dict,orient = 'index')
  ratings_df = ratings_df.drop(columns = [str(season)+'League' for season in range(fy,2020)])
   
  vegas = vegas[~vegas.isna().any(axis=1)]  
  m = len(vegas)

  num = sum((vegas.my_pred-vegas.vegas_spread)**2)/m+sum((vegas.vegas_spread-vegas.actual_spread)**2)/m-sum((
    vegas.my_pred-vegas.actual_spread)**2)/m
  den = math.sqrt(sum((vegas.my_pred-vegas.vegas_spread)**2)/m)*math.sqrt(
    sum((vegas.vegas_spread-vegas.actual_spread)**2)/m+sum((vegas.my_pred-vegas.actual_spread)**2)/m)
  inner = (num/den)*math.sqrt(m/2)
  
  print(norm.cdf(inner))
  print(ratings_df.sort_values('2019Rating',ascending=False).head(35).to_string())
  #print(vegas.sample(frac=1).tail(40).to_string())

  diff = vegas[abs(vegas.vegas_spread - vegas.my_pred) > 10].copy()
  #print(diff.sample(frac=1).head(40).to_string())
  print('High vegas vs model difference',
        sum(abs(diff.vegas_error))/len(diff),sum(abs(diff.my_error))/len(diff))

  low_diff = vegas[abs(vegas.vegas_spread - vegas.my_pred) < 2].copy()
  #print(low_diff.sample(frac=1).head(40).to_string())
  print('Low vegas vs model difference',
        sum(abs(low_diff.vegas_error))/len(low_diff),sum(abs(low_diff.my_error))/len(low_diff))

  low = vegas[abs(vegas.vegas_spread - vegas.actual_spread) < 2].copy()
  #print(low.sample(frac=1).head(40).to_string())
  print('Low vegas error',
        sum(abs(low.vegas_error))/len(low),sum(abs(low.my_error))/len(low))
  #print(len(diff),len(low_diff),len(low))

  low_my = vegas[abs(vegas.my_pred - vegas.actual_spread) < 2].copy()
  #print(low.sample(frac=1).head(40).to_string())
  print('Low model error',
    sum(abs(low_my.vegas_error))/len(low_my),sum(abs(low_my.my_error))/len(low_my))

print(tot_error/n)"""
