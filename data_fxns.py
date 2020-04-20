import pandas as pd
import numpy as np
from itertools import accumulate as acc
import data_scrape
import neural_network
import cfg


def data_gather(first_season, last_season, data_type = 'adv', verbose = True):  
  """
  Returns a DataFrame of cumulative statistics that the model can be trained on.
  ----------  
  Parameters
  ----------
    first_season, last_season: int
      The function gathers statistics from the years in the range [first_season, last_season].
    data_type: str
      The type of statistics to be prepared. Right now, the function only works for "advanced" statistics.
      I plan to add the ability to gather and prepare "regular" statistics as well.
    verbose: bool
  -----
  Notes
  -----
  The function performs several tasks:
   - Scrapes game information, per-game statistics, and talent ratings
   - Calculates each team's cumulative per-game statistics
     - e.g. a team's offensive efficiency statistic in its 5th game will be the average of its offensive efficiency statistics
       from games 1-4.
   - Normalizes the talent ratings and cumulative statistics
   - Initializes SOS and last season ratings
   - It is admittedly messy
  """
  total = pd.DataFrame()
  for season in range(first_season, last_season + 1):
    if verbose == True:
      print(season)

    games = data_scrape.games_scrape(season)
    games = game_data_filter(games, season)
    game_data = pd.DataFrame({'id': games.id,
                              'home_team': games.home_team,
                              'away_team': games.away_team,
                              'home_conference': games.home_conference,
                              'away_conference': games.away_conference,
                              'season': games.season,
                              'week': games.week,
                              'neutral': games.neutral_site,
                              'y_actual': games.home_points - games.away_points,
                              'home_points': games.home_points,
                              'away_points': games.away_points,
                              'home_opponents': None,
                              'away_opponents': None
                             }
                            )

    high_score = max(max(game_data.home_points), max(game_data.away_points))
    game_data.home_points /= high_score
    game_data.away_points /= high_score

    if data_type == 'adv':
      season_data = data_scrape.adv_data_scrape(season)
      season_data = adv_season_data_filter(season_data, season).drop(
        columns = ['gameId']).sort_values('week').fillna(0)
    elif data_type == 'reg':
      season_data = data_scrape.reg_data_scrape(season).drop(
        columns = ['gameId']).sort_values('week').fillna(0)
      
    for col in season_data.columns[3:]:
      season_data[col] -= min(season_data[col])
      season_data[col] /= max(season_data[col])
    n_cols = len(season_data.columns)
    
    season_data['offpoints'] = None
    season_data['defpoints'] = None
    season_data['games'] = None

    fbs_teams = set([team for team in season_data.team if len(season_data[season_data.team == team]) > 3])
    
    for team in fbs_teams:
      season_data = season_data.append(
        pd.Series([20,team,team]+[None]*n_cols, index = season_data.columns),
        ignore_index = True)

      team_season_data = season_data[season_data.team==team]
      opponents = list(team_season_data.opponent)
      game_data = game_data.append(
        pd.Series([None]+[team]*2+[None]*2+[season, 20]+[None]*4+[opponents]+[None],
        index = game_data.columns), ignore_index=True)

      n_games = len(team_season_data) - 1
      game_range_discount = np.array(list(acc([1]*(n_games), lambda x, y: x*cfg.r + y)))
      tsd_index = team_season_data.index      
      for col in season_data.columns[3:-3]:
        a = team_season_data[col][:-1]
        season_data.at[tsd_index, col] = pd.Series(np.concatenate([[0.5],
            np.array(list(acc(a, lambda x, y: x*cfg.r + y)))/game_range_discount]),
          index = tsd_index)
        
      season_data.at[tsd_index, 'games'] = pd.Series([i/n_games for i in range(n_games + 1)],
          index = tsd_index)
      
      season_data.at[tsd_index, 'offpoints'] = pd.Series(np.concatenate([[0.5], np.array(
        list(acc(game_data.loc[game_data.home_team == team, 'home_points'].append(
          game_data.loc[game_data.away_team == team, 'away_points']
          ).dropna().sort_index()[:len(tsd_index) - 1], lambda x, y: x*cfg.r + y))
          )/game_range_discount]),
        index = tsd_index)
      
      season_data.at[tsd_index, 'defpoints'] = pd.Series(np.concatenate([[0.5], np.array(
        list(acc(game_data.loc[game_data.home_team == team, 'away_points'].append(
          game_data.loc[game_data.away_team == team, 'home_points']
          ).dropna().sort_index()[:len(tsd_index) - 1], lambda x, y: x*cfg.r + y))
          )/game_range_discount]),
        index = tsd_index)

    talent = data_scrape.talent_scrape(season)
    talent.talent = talent.talent.astype('float32')
    talent.talent -= min(talent.talent)
    talent.talent /= max(talent.talent)
    season_data = season_data.merge(talent, how = 'left', on = 'team').fillna(0.0)
    
    game_data = game_data.merge(season_data, left_on = ['home_team', 'away_team', 'week'],
                                right_on = ['team', 'opponent', 'week']
                                ).drop(columns = ['team', 'opponent'])
    col_dict = {}
    for col in game_data.columns[13:]:
      col_dict[col] = 'home_' + col
    game_data = game_data.rename(columns = col_dict)
    
    game_data['home_last_rating'] = 0.5
    game_data['home_SOS'] = 0.45

    game_data = game_data.merge(season_data, left_on = ['away_team', 'home_team', 'week'],
                                right_on = ['team', 'opponent', 'week']
                               ).drop(columns = ['team', 'opponent'])
    col_dict = {}
    for col in game_data.columns[(len(game_data.columns) - 13)//2 + 14:]:
      col_dict[col] = 'away_' + col
    game_data = game_data.rename(columns = col_dict)

    game_data['away_last_rating'] = 0.5
    game_data['away_SOS'] = 0.45

    game_data = add_sp_ratings(game_data, season)

    total = pd.concat([total, game_data])
    
  return total.reset_index(drop=True)


def add_sp_ratings(game_data, season):
  sp = data_scrape.sp_scrape(season - 1)
  sp.rating -= min(sp.rating)
  sp.rating /= max(sp.rating)
  for i in range(len(sp)):
    team, rating = sp.loc[i, 'team'], sp.loc[i, 'rating']
    game_data.loc[(game_data.home_team == team)&(game_data.season == season), 'home_last_rating'] = rating
    game_data.loc[(game_data.away_team == team)&(game_data.season == season), 'away_last_rating'] = rating

  return game_data


def ratings_init(game_data):
  """
  Initializes the ratings dictionary that stores a team's ratings and division (FBS or FCS)
  for each season in [first_season, last_season]
  """
  teams = set(list(game_data.home_team)+list(game_data.away_team))

  for team in teams:
    cfg.ratings_dict.setdefault(team, {})
    for season in range(cfg.first_season, cfg.last_season+1):
      if len(game_data[((game_data.home_team == team)|(game_data.away_team == team))&
                       (game_data.season == season)]) > 3:
        cfg.ratings_dict[team][str(season) + 'League'] = 'FBS'
        cfg.ratings_dict[team][str(season) + 'Rating'] = 0.5
      else:
        cfg.ratings_dict[team][str(season) + 'League'] = 'FCS'
        cfg.ratings_dict[team][str(season) + 'Rating'] = 0.15         


def data_init(game_data, pca_components):
  """
  Initializes the game_data for training on [cfg.first_season, cfg.last_season]
  ----------  
  Parameters
  ----------
    game_data: DataFrame
    pca_components: bool, float, or int
      If float or int, performs Principal Component Analysis on game_data
  -----
  Notes
  -----
   - Constrains the data to within [first_season, last_season]
   - Scrapes S&P+ ratings from first_season - 1 and normalizes them
   - Sets the teams' last_ratings for first_season equal to their S&P+ ratings from first_season - 1
   - Shuffles game_data
  """
  if type(pca_components) == float or type(pca_components) == int:
    game_data = pca(game_data, pca_components)
    
  game_data = game_data[(game_data.season >= cfg.first_season)&(game_data.season <= cfg.last_season)].copy()
    
  """sp = data_scrape.sp_scrape(cfg.first_season - 1)
  sp.rating -= min(sp.rating)
  sp.rating /= max(sp.rating)
  for i in range(len(sp)):
    team, rating = sp.loc[i, 'team'], sp.loc[i, 'rating']
    game_data.loc[(game_data.home_team == team)&(game_data.season == cfg.first_season), 'home_last_rating'] = rating
    game_data.loc[(game_data.away_team == team)&(game_data.season == cfg.first_season), 'away_last_rating'] = rating
  """ 
  return game_data.sample(frac = 1).reset_index(drop = True)


def pca(game_data, n_components):
  """
  Performs PCA on game_data and returns the transformed data.
  ----------  
  Parameters
  ----------
    game_data: DataFrame
    pca_components: bool, float, or int
      If float or int, performs Principal Component Analysis on game_data  
  """
  from sklearn.decomposition import PCA
  game_data = game_data.drop(columns = ['home_SOS', 'home_last_rating',
                                        'away_SOS', 'away_last_rating'])
  
  n_cols = (len(game_data.columns) - 13)//2
  game_info = game_data.loc[:,game_data.columns[:13]]
  home_raw = game_data.loc[:,game_data.columns[13:n_cols + 13]]
  away_raw = game_data.loc[:,game_data.columns[n_cols + 13:]].copy()
  away_raw.columns = home_raw.columns
  temp = pd.concat([home_raw, away_raw], axis = 0)
  pca = PCA(n_components = n_components)
  pca.fit(temp)
  
  home_away = pd.DataFrame(pca.transform(temp))
  home = home_away.iloc[:len(home_raw)].copy()
  home['home_SOS'] = 0.45
  home['home_last_rating'] = 0.5
  
  away = home_away.iloc[len(away_raw):].copy().reset_index(drop = True)
  away.columns = [i for i in range(len(home.columns) - 2, len(home.columns) + len(away.columns) - 2)]
  away['away_SOS'] = 0.45
  away['away_last_rating'] = 0.5
  
  return game_info.join(home).join(away)


def index_dict_init(game_data):
  """
  Initializes the index_dict, which improves the lookup speed of a team's games
  Relevant for train_fxns.ratings_calc and train_fxns.sos_calc
  """
  for team in set(list(game_data.home_team) + list(game_data.away_team)):
    cfg.index_dict[team] = {}
    for season in range(cfg.first_season, cfg.last_season+1):
      home_indices = game_data[(game_data.season == season)&(game_data.home_team == team)].sort_values('week').index
      away_indices = game_data[(game_data.season == season)&(game_data.away_team == team)].sort_values('week').index
      total_indices = game_data[(game_data.season == season)&(
        (game_data.home_team == team)|(game_data.away_team == team))].sort_values('week').index
      cfg.index_dict[team][season] = {'home': home_indices,
                                      'away': away_indices,
                                      'total': total_indices}


def nn_list_init(learn_rate, season_discount, tol, window):
  """
  Populates the list of neural networks that will be trained.
  """
  for i in range(1,14):
    cfg.nn_list.append(neural_network.NeuralNet(i, window, learn_rate, season_discount, tol))


def custom_train_test_split(game_data, train_size, first_week, last_week): 
  """
  Splits the game data into (shuffled) train and test sets.
  ----------  
  Parameters
  ----------
    game_data: DataFrame
      The total data from which the train and test sets are drawn
    train_size: float
      The proportion of the relevant data to be used for training
    first_week, last_weeks: int
      The first and last weeks of the season from which data is drawn
  """  
  game_data_range = game_data[
    (game_data.week >= first_week)&(game_data.week <= min(19,last_week))
    ].reset_index(drop=True)

  train = game_data_range.iloc[:int(len(game_data_range)*train_size)
    ].sample(frac=1).reset_index(drop=True) 
  test = game_data_range.iloc[int(len(game_data_range)*train_size):
    ].reset_index(drop=True)

  return train, test


def game_data_filter(game_data, season):
  """
  A function that fixes a problem with the 2012 game data
  ----------
  Parameters
  ----------
    game_data: DataFrame
    season: int
  """
  if season == 2012:
    army_temple = game_data[(game_data.home_team == 'Army')&
                            (game_data.away_team == 'Temple')]
    if len(army_temple) > 1:
      game_data = game_data.drop(
        index = army_temple.index[0]).reset_index(drop=True)          
      
  return game_data


def adv_season_data_filter(season_data, season):
  """
  A function that fixes a problem with the advanced data from multiple years
  ----------
  Parameters
  ----------
    season_data: DataFrame
    season: int
  """
  if season == 2013:
    t = [333332117, 333060070]
    season_data = season_data.drop(index = season_data[season_data.gameId.isin(t)].index)
  elif season == 2014:
    season_data.loc[(season_data.team == 'Stanford')&(season_data.opponent == 'UC Davis'),
                    'offense.drives'] = 14
    season_data.loc[(season_data.team == 'Stanford')&(season_data.opponent == 'UC Davis'),
                    'defense.drives'] = 15
  elif season == 2015:
    season_data.loc[(season_data.team == 'Alabama')&(season_data.opponent == 'Charleston Southern'),
                    'offense.drives'] = 10
    season_data.loc[(season_data.team == 'Alabama')&(season_data.opponent == 'Charleston Southern'),
                    'defense.drives'] = 11
  elif season == 2016:
    season_data.loc[(season_data.team == 'Kent State')&(season_data.opponent == 'Northern Illinois'),
                    'offense.drives'] = 16
    season_data.loc[(season_data.team == 'Kent State')&(season_data.opponent == 'Northern Illinois'),
                    'defense.drives'] = 17
    season_data.loc[(season_data.team == 'Northern Illinois')&(season_data.opponent == 'Kent State'),
                    'offense.drives'] = 17
    season_data.loc[(season_data.team == 'Northern Illinois')&(season_data.opponent == 'Kent State'),
                    'defense.drives'] = 16
    
  return season_data


#cfg.init(0.0,2006,2019)
#import sys
#data_gather(2006,2019).to_pickle(
 # r'C:\Users\Visitor\AppData\Local\Programs\Python\Python38-32\cfbdata\cfb619_advseasondata.pkl')
#sys.exit()
