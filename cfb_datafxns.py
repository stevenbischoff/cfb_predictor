import pandas as pd
import numpy as np
from cfb_datascrape import *
import time


def data_gather(first_season, last_season, data_type = 'adv'):

  tot = pd.DataFrame()

  for season in range(first_season, last_season+1):
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

    season_cols = season_data.columns
    n_cols = len(season_cols)-3
    game_cols = game_data.columns
    for team in fbs_teams:
      season_data = season_data.append(
        pd.Series([20,team,team]+[None]*n_cols,index=season_cols),
        ignore_index=True)

      team_season_data = season_data[season_data.team==team]
      opponents = list(team_season_data['opponent'])

      game_data = game_data.append(
        pd.Series([team,team]+[None]*2+[season, 20]+[None]*4+[opponents]+[None],
        index=game_cols),
        ignore_index=True)
      
      for col in season_data.columns[3:]:
        a = team_season_data[col][:-1]
        season_data.loc[team_season_data.index,col] = pd.Series(
          np.concatenate([[0.5],np.cumsum(a)/np.arange(1,len(a)+1)]),
          team_season_data.index, dtype='float32')

    talent = talent_scrape(season)
    talent.talent = talent.talent.astype('float32')
    try:
      talent.talent /= max(talent.talent)
    except:
      talent.talent = np.nan
    season_data = season_data.merge(talent,how='left', on='team').fillna(0.5)
    game_data = game_data.merge(season_data,
                                left_on = ['home_team','away_team','week'],
                                right_on = ['team','opponent','week']
                                ).drop(columns = ['team','opponent'])
    for col in game_data.columns[12:]:
      game_data = game_data.rename(columns = {col:'home_'+col})

    game_data = game_data.merge(season_data,
                                left_on = ['away_team','home_team','week'],
                                right_on = ['team','opponent','week']
                               ).drop(columns = ['team','opponent'])
    for col in game_data.columns[(len(game_data.columns)-12)//2 + 12:]:
      game_data = game_data.rename(columns = {col:'away_'+col})

    tot = pd.concat([tot,game_data])
    
  return tot


def data_init(game_data, first_season, last_season):

  game_data = game_data[(game_data.season >= first_season)&
                        (game_data.season <= last_season)]
  
  game_data.insert((len(game_data.columns)-12)//2 + 12, 'home_last_rating', 0.5)
  game_data.insert((len(game_data.columns)-12)//2 + 12, 'home_SOS', 0.45)
  game_data.insert(len(game_data.columns), 'away_last_rating', 0.5)
  game_data.insert(len(game_data.columns), 'away_SOS', 0.45)

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
      if len(game_data_season[game_data_season.home_team==team])>2 or len(
        game_data_season[game_data_season.away_team==team])>3:
        sos_list.append(0.45)
        rating_list.append(0.5)
      else:
        sos_list.append('FCS')
        rating_list.append(0.15)
    sos[str(season)+'SOS'] = sos_list
    sos[str(season)+'Rating'] = rating_list
    
  game_data = game_data.sample(frac=1).reset_index(drop=True)
  
  return game_data, sos


def custom_train_test_split(game_data, train_size, first_season, 
                            last_season, first_week, last_week):
  game_data_range = game_data[
    ((game_data.week>=first_week)&(game_data.week<=min(19,last_week)))&
    (game_data.season>=first_season)&(game_data.season<=last_season)
    ].reset_index(drop=True)

  train = game_data_range.iloc[:int(len(game_data_range)*train_size)
    ].sample(frac=1).reset_index(drop=True) 
  test = game_data_range.iloc[int(len(game_data_range)*train_size):
    ].reset_index(drop=True)

  return train, test

def games_filter(game_data, season):
  if season == 2012:
    army_temple = game_data[(game_data.home_team == 'Army')&
                            (game_data.away_team == 'Temple')]
    if len(army_temple) > 1:
      game_data = game_data.drop(
        index = army_temple.index[0]).reset_index(drop=True)          
      
  
  """team_game_data = game_data[(game_data.home_team == team)|
                             (game_data.away_team == team)]
  for week in team_game_data.week:
    if len(team_game_data[team_game_data.week == week]) > 1:
      week_data = team_game_data[team_game_data.week == week]
      if len(set(list(week_data.home_team)+list(week_data.away_team))) == 2:
        game_data = game_data.drop(index = week_data.index[0]).reset_index(drop=True)

      elif week == 1:
        print(team, 'week 0')
        game_data.loc[week_data.index[0],'week'] = 0
        team_game_data = game_data[(game_data.home_team == team)|
                                   (game_data.away_team == team)]
      else:
        print(team, 'last')
        game_data.loc[week_data.index[-1],'week'] += 1
        team_game_data = game_data[(game_data.home_team == team)|
                                   (game_data.away_team == team)]"""
  return game_data
