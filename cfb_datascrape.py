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
  Returns a DataFrame with 247 composite talent ratings for a season, which only go back through 2015.
  """
  
  
  base_url = 'https://api.collegefootballdata.com/talent'  
  parameters = {'year':season}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    return pd.DataFrame(response.json(),columns = ['school','talent']
        ).rename(columns = {'school':'team'})
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))     
