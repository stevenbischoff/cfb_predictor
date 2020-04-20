import requests
import pandas as pd


def adv_data_scrape(season):  
  """
  Returns advanced game stats for a season pulled using the collegefootballdata API. 
  The function concatenates regular season and postseason stats into a single Pandas DataFrame.  
  """
  base_url = 'https://api.collegefootballdata.com/stats/game/advanced'    
  parameters = {'year':season,'seasonType':'regular'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    reg_df = pd.json_normalize(response.json())
    last_week = max(reg_df.week)
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
 
  parameters = {'year':season,'seasonType':'postseason'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    post_df = pd.json_normalize(response.json())
    post_df.week += last_week
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
    
  return pd.concat([reg_df,post_df])


def games_scrape(season):  
  """
  Returns a DataFrame of game information for a season that includes the conferences of the participants,
  whether the game was played on a neutral field, and the final score, all of which are relevant for downstream
  data manipulation and training.
  Also concatenates regular season and postseason information.
  """
  base_url = 'https://api.collegefootballdata.com/games'
  parameters = {'year':season,'seasonType':'regular'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    reg_games_df = pd.json_normalize(response.json())
    last_week = max(reg_games_df.week)
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
 
  parameters = {'year':season,'seasonType':'postseason'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    post_games_df = pd.json_normalize(response.json())
    post_games_df.week += last_week
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
    
  return pd.concat([reg_games_df,post_games_df])


def talent_scrape(season):  
  """
  Returns a DataFrame with team talent ratings for the season.
   - For seasons 2015+, returns the 247 composite talent ratings for that season.
   - For seasons before 2015 the composite is unavailable, so the function returns the sum of the recruiting class
     ratings for the past four years. 
  """  
  if season >= 2015:
    base_url = 'https://api.collegefootballdata.com/talent'  
    parameters = {'year':season}
    
    response = requests.get(base_url,params=parameters)
    if response.status_code == 200:
      return pd.DataFrame(response.json(),columns = ['school', 'talent']
          ).rename(columns = {'school':'team'})
    else:
      raise Exception('Request failed with status code: '+str(response.status_code))
  else:
    tot = pd.DataFrame(columns = ['team'])
    for temp_season in range(season - 3, season + 1):
      base_url = 'https://api.collegefootballdata.com/recruiting/teams'
      parameters = {'year':temp_season}
    
      response = requests.get(base_url,params=parameters)
      if response.status_code == 200:
        season_recruiting = pd.DataFrame(response.json(), columns = ['team', 'points']
            ).rename(columns = {'points':str(temp_season) + 'points'})
      else:
        raise Exception('Request failed with status code: '+str(response.status_code))
      tot = tot.merge(season_recruiting, how = 'right', on = 'team').fillna(0)
      tot[str(temp_season) + 'points'] = tot[str(temp_season) + 'points'].astype('float32')
      
    tot['talent'] = sum([tot[col] for col in tot.columns[1:]])
    tot = tot.drop(columns = tot.columns[1:-1])
    
    return tot


def sp_scrape(season):  
  """
  Returns a DataFrame with S&P+ Ratings from a season
  """
  base_url = 'https://api.collegefootballdata.com/ratings/sp'
  parameters = {'year':season}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    sp = pd.DataFrame(response.json(), columns = ['team','rating'])
    return sp.drop(index = sp.index[-1])
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
