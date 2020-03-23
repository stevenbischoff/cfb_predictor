import pandas as pd
import numpy as np
from cfb_datascrape import *


def data_gather(first_season, last_season, data_type = 'adv', verbose = True):
  
  """
  Returns a DataFrame of cumulative statistics that the model can be trained on.
  ----------  
  Parameters
  ----------
    first_season, last_season: int
      The function gathers statistics from the years in the range [first_year, last_year]
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

  tot = pd.DataFrame()

  for season in range(first_season, last_season + 1):
    if verbose == True:
      print(season)

    games = games_scrape(season)
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
      season_data = adv_data_scrape(season).drop(columns = ['gameId']).sort_values(
        'week').fillna(0)
    elif data_type == 'reg':
      season_data = reg_data_scrape(season).drop(columns = ['gameId']).sort_values(
        'week').fillna(0)
      
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
      n_games = len(team_season_data) - 1
      tsd_index = team_season_data.index
      opponents = list(team_season_data.opponent)

      game_data = game_data.append(
        pd.Series([team,team]+[None]*2+[season, 20]+[None]*4+[opponents]+[None], index = game_cols),
        ignore_index=True)

      game_range = np.arange(1, n_games+1)      
      for col in season_cols[3:-1]:
        a = team_season_data[col][:-1]
        season_data.loc[tsd_index, col] = pd.Series(np.concatenate([[0.5], np.cumsum(a)/game_range]),
          index = tsd_index)
        
      season_data.loc[tsd_index, 'games'] = pd.Series([i/n_games for i in range(n_games + 1)],
          index = tsd_index)

    talent = talent_scrape(season)
    talent.talent = talent.talent.astype('float32')
    try:
      talent.talent /= max(talent.talent)
    except:
      talent.talent = np.nan
    season_data = season_data.merge(talent, how = 'left', on = 'team').fillna(0.5)
    
    game_data = game_data.merge(season_data, left_on = ['home_team','away_team','week'],
                                right_on = ['team','opponent','week']
                                ).drop(columns = ['team','opponent'])
    for col in game_data.columns[12:]:
      game_data = game_data.rename(columns = {col:'home_'+col})

    game_data = game_data.merge(season_data, left_on = ['away_team','home_team','week'],
                                right_on = ['team','opponent','week']
                               ).drop(columns = ['team','opponent'])
    for col in game_data.columns[(len(game_data.columns)-12)//2 + 12:]:
      game_data = game_data.rename(columns = {col:'away_'+col})

    tot = pd.concat([tot,game_data])
     
  tot.insert((len(tot.columns)-12)//2 + 12, 'home_last_rating', 0.5)
  tot.insert((len(tot.columns)-12)//2 + 12, 'home_SOS', 0.45)
  tot.insert(len(tot.columns), 'away_last_rating', 0.5)
  tot.insert(len(tot.columns), 'away_SOS', 0.45)
  
  tot = tot.sample(frac=1).reset_index(drop=True)
    
  return tot


def sos_init(game_data, first_season, last_season):
  
  """
  Initializes the SOS DataFrame, which stores end-of-season ratings and is used for SOS calculations
  ----------  
  Parameters
  ----------
    game_data: DataFrame
    first_season, last_season: int
  """

  teams = list(set(list(game_data.home_team)+list(game_data.away_team)))

  sos = pd.DataFrame(index=[i for i in range(len(teams))],
                     columns=['Team']+[
                       str(season)+'SOS' for season in range(first_season, last_season+1)]+[
                         str(season)+'Rating' for season in range(first_season, last_season+1)],
                     dtype = 'float32')

  sos['Team'] = teams
  for season in range(first_season, last_season+1):
    game_data_season = game_data[game_data.season == season]
    sos_list = []
    rating_list = []
    for team in teams:
      if len(game_data_season[game_data_season.home_team == team]) > 2 or len(
        game_data_season[game_data_season.away_team == team]) > 3:
        sos_list.append(0.45)
        rating_list.append(0.5)
      else:
        sos_list.append('FCS')
        rating_list.append(0.15)
    sos[str(season)+'SOS'] = sos_list
    sos[str(season)+'Rating'] = rating_list
  
  return sos


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
