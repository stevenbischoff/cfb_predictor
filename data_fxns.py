import pandas as pd
import numpy as np
import data_scrape
import train_fxns
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
  """
  total = pd.DataFrame()
  for season in range(first_season, last_season + 1):
    if verbose == True:
      print(season)

    games = data_scrape.games_scrape(season)
    games = games_filter(games, season)
    game_data = pd.DataFrame({'home_team':games.home_team,
                              'away_team':games.away_team,
                              'home_conference':games.home_conference,
                              'away_conference':games.away_conference,
                              'season':games.season,
                              'week':games.week,
                              'neutral':games.neutral_site,
                              'y_actual':games.home_points - games.away_points,
                              'home_points':games.home_points,
                              'away_points':games.away_points,
                              'home_opponents':None,
                              'away_opponents':None
                             }
                            )

    if data_type == 'adv':
      season_data = data_scrape.adv_data_scrape(season).drop(
        columns = ['gameId']).sort_values('week').fillna(0)
    elif data_type == 'reg':
      season_data = data_scrape.reg_data_scrape(season).drop(
        columns = ['gameId']).sort_values('week').fillna(0)
      
    for col in season_data.columns[3:]:
      season_data[col] -= min(season_data[col])
      season_data[col] /= max(season_data[col])

    fbs_teams = set([team for team in season_data.team if len(season_data[season_data.team == team]) > 3])

    season_cols = list(season_data.columns) + ['games']
    n_cols = len(season_cols) - 3
    game_cols = game_data.columns
    for team in fbs_teams:
      season_data = season_data.append(
        pd.Series([20,team,team]+[None]*n_cols, index = season_cols),
        ignore_index = True)

      team_season_data = season_data[season_data.team==team]
      opponents = list(team_season_data.opponent)
      game_data = game_data.append(
        pd.Series([team]*2+[None]*2+[season, 20]+[None]*4+[opponents]+[None], index = game_cols),
        ignore_index=True)

      n_games = len(team_season_data) - 1
      game_range = np.arange(1, n_games+1)
      tsd_index = team_season_data.index      
      for col in season_cols[3:-1]:
        a = team_season_data[col][:-1]
        season_data.at[tsd_index, col] = pd.Series(np.concatenate([[0.5], np.cumsum(a)/game_range]),
          index = tsd_index)
        
      season_data.at[tsd_index, 'games'] = pd.Series([i/n_games for i in range(n_games + 1)],
          index = tsd_index)

    talent = data_scrape.talent_scrape(season)
    talent.talent = talent.talent.astype('float32')
    talent.talent -= min(talent.talent)
    talent.talent /= max(talent.talent)
    season_data = season_data.merge(talent, how = 'left', on = 'team').fillna(0.5)
    
    game_data = game_data.merge(season_data, left_on = ['home_team','away_team','week'],
                                right_on = ['team','opponent','week']
                                ).drop(columns = ['team','opponent'])
    col_dict = {}
    for col in game_data.columns[12:]:
      col_dict[col] = 'home_' + col
    game_data = game_data.rename(columns = col_dict)

    game_data = game_data.merge(season_data, left_on = ['away_team','home_team','week'],
                                right_on = ['team','opponent','week']
                               ).drop(columns = ['team','opponent'])
    col_dict = {}
    for col in game_data.columns[(len(game_data.columns) - 12)//2 + 12:]:
      col_dict[col] = 'away_' + col
    game_data = game_data.rename(columns = col_dict)

    total = pd.concat([total,game_data])
     
  total.insert((len(total.columns)-12)//2 + 12, 'home_last_rating', 0.5)
  total.insert((len(total.columns)-12)//2 + 12, 'home_SOS', 0.45)
  total.insert(len(total.columns), 'away_last_rating', 0.5)
  total.insert(len(total.columns), 'away_SOS', 0.45)
    
  return total


def ratings_init(game_data, first_season, last_season):
  """
  Initializes the ratings dictionary that stores a team's ratings and whether they are in FBS or FCS
  for each season in [first_season, last_season]
  """
  teams = set(list(game_data.home_team)+list(game_data.away_team))

  for team in teams:
    cfg.ratings_dict.setdefault(team, {})
    for season in range(first_season, last_season+1):
      if len(game_data[((game_data.home_team == team)|(game_data.away_team == team))&
                       (game_data.season == season)]) > 3:
        cfg.ratings_dict[team][str(season) + 'League'] = 'FBS'
        cfg.ratings_dict[team][str(season) + 'Rating'] = 0.5
      else:
        cfg.ratings_dict[team][str(season) + 'League'] = 'FCS'
        cfg.ratings_dict[team][str(season) + 'Rating'] = 0.15         


def data_init(game_data, first_season, last_season):
  """
  Initializes the game_data for training on [first_season, last_season]
  ----------  
  Parameters
  ----------
    game_data: DataFrame
    first_season, last_season: int
  -----
  Notes
  -----
   - Constrains the data to within [first_season, last_season]
   - Sets the teams' last_ratings for first_season equal to their S&P+ ratings from first_season - 1
   - Shuffles game_data
  """
  game_data = game_data.loc[(game_data.season >= first_season)&(game_data.season <= last_season),:].copy()

  sp = sp_scrape(first_season - 1)
  sp.rating -= min(sp.rating)
  sp.rating /= max(sp.rating)
  for i in range(len(sp)):
    team, rating = sp.loc[i, 'team'], sp.loc[i, 'rating']
    game_data.loc[(game_data.home_team == team)&(game_data.season == first_season), 'home_last_rating'] = rating
    game_data.loc[(game_data.away_team == team)&(game_data.season == first_season), 'away_last_rating'] = rating

  return game_data.sample(frac = 1).reset_index(drop = True)


def index_dict_init(game_data, first_season, last_season):
  for team in set(list(game_data.home_team) + list(game_data.away_team)):
    cfg.index_dict[team] = {}
    for season in range(first_season, last_season+1):
      home_indices = game_data[(game_data.season == season)&(game_data.home_team == team)].sort_values('week').index
      away_indices = game_data[(game_data.season == season)&(game_data.away_team == team)].sort_values('week').index
      total_indices = game_data[(game_data.season == season)&(
        (game_data.home_team == team)|(game_data.away_team == team))].sort_values('week').index
      cfg.index_dict[team][season] = {'home': home_indices,
                                  'away': away_indices,
                                  'total': total_indices}


def nn_list_init(n_cols, learn_rate, season_discount, tol, window):    
  for i in range(1,14):
    cfg.nn_list.append(train_fxns.NeuralNet(n_cols, i, window, learn_rate, season_discount, tol))


def custom_train_test_split(game_data, train_size, first_week, last_week): 
  """
  Splits the game data into (shuffled) train and test sets
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
    (game_data.week>=first_week)&(game_data.week<=min(19,last_week))
    ].reset_index(drop=True)

  train = game_data_range.iloc[:int(len(game_data_range)*train_size)
    ].sample(frac=1).reset_index(drop=True) 
  test = game_data_range.iloc[int(len(game_data_range)*train_size):
    ].reset_index(drop=True)

  return train, test


def games_filter(game_data, season):
  """
  A function that fixes a problem with the 2012 advanced data
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
