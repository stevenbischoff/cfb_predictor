import numpy as np
import pandas as pd
import copy
from itertools import accumulate as acc
import cfb_predictor.cfg as cfg


def ratings_sos_calculation(nn, game_data, n_rounds):
  """
  Performs a specified number of ratings and SOS updates on game_data.
  ----------
  Parameters
  ----------
    nn: NeuralNet
    game_data: DataFrame
    n_rounds: int
  """
  for n in range(n_rounds):
    ratings_calculation(nn, game_data)
    sos_calculation(game_data)

    
def ratings_calculation(nn, game_data):
  """
  Calculates the final ratings for every FBS team and updates the last season ratings in game_data 
  Each team's final statistics are stored in a "week 20" row of game_data
  ----------
  Parameters
  ----------
    nn: NeuralNet
    game_data: DataFrame
  """
  game_data1 = game_data[game_data.week == 20]
  np.apply_along_axis(team_rating, 1, game_data1, nn, game_data)


def team_rating(tg, nn, game_data):
  """
  Calculates a team's final rating for a particular season. Updates the SOS dataframe with this information, and also
  updates the last_rating information in game_data iff the season is less than last_season
  ----------
  Parameters
  ----------
    tg: numpy array
    nn: NeuralNet
    game_data: DataFrame
  """
  team = tg[1]
  season = tg[5]

  s1 = nn.feedforward(tg[13:cfg.n_cols+13].astype('float32'))  
  cfg.ratings_dict[team][str(season)+'Rating'] = s1
  if season < cfg.last_season:
    season_indices = cfg.index_dict[team][season+1]
    game_data.at[season_indices['home'], 'home_last_rating'] = s1
    game_data.at[season_indices['away'], 'away_last_rating'] = s1


def sos_calculation(game_data):
  """
  For each team, performs Strength of Schedule (SOS) calculations for each season for which they were in the FBS
  This function takes up a majority of training time, and I'm always looking for ways to improve its speed.
  """
  for team in cfg.ratings_dict.keys():
    for season in range(cfg.first_season, cfg.last_season + 1):
      if cfg.ratings_dict[team][str(season)+'League'] != 'FCS':
        season_indices = cfg.index_dict[team][season]
        opponents = game_data.at[season_indices['total'][-1],'home_opponents'][:-1]
        opp_ratings = np.array([cfg.ratings_dict[opp][str(season) + 'Rating'] for opp in opponents])
        avg_sos = pd.Series(np.concatenate([[0.5],
            np.array(list(acc(opp_ratings, lambda x, y: x*cfg.r + y)))/
            np.array(list(acc([1]*(len(opp_ratings)), lambda x, y: x*cfg.r + y)))]),
          season_indices['total'])
        
        game_data.at[season_indices['home'], 'home_SOS'] = avg_sos.loc[season_indices['home']]
        game_data.at[season_indices['away'], 'away_SOS'] = avg_sos.loc[season_indices['away']]
