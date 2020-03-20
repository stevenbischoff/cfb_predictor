import requests
import pandas as pd

def adv_data_scrape(year):

  base_url = 'https://api.collegefootballdata.com/stats/game/advanced'    
  parameters = {'year':year,'seasonType':'regular'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    reg_df = pd.json_normalize(response.json())
    last_week = max(reg_df.week)
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
 
  parameters = {'year':year,'seasonType':'postseason'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    post_df = pd.json_normalize(response.json())
    post_df.week += last_week
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
    
  return pd.concat([reg_df,post_df])

def games_scrape(year):

  base_url = 'https://api.collegefootballdata.com/games'
  parameters = {'year':year,'seasonType':'regular'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    reg_games_df = pd.json_normalize(response.json())
    last_week = max(reg_games_df.week)
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
 
  parameters = {'year':year,'seasonType':'postseason'}
  
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    post_games_df = pd.json_normalize(response.json())
    post_games_df.week += last_week
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))
    
  return pd.concat([reg_games_df,post_games_df])

def talent_scrape(year):
  base_url = 'https://api.collegefootballdata.com/talent'
  
  parameters = {'year':year}
  response = requests.get(base_url,params=parameters)
  if response.status_code == 200:
    return pd.DataFrame(response.json(),columns = ['school','talent']
        ).rename(columns = {'school':'team'})
  else:
    raise Exception('Request failed with status code: '+str(response.status_code))

  if year == first_year:
    tot_talent_df = year_talent_df
  else:
    tot_talent_df = tot_talent_df.merge(year_talent_df,on='team')

  return tot_talent_df

      
