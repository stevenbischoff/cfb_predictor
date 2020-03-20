import pandas as pd
import numpy as np
import math
import copy
from cfb_trainfxns_1 import *
from cfb_trainingdataprep_1 import *
from cfb_dataprep_1 import *

import sys
import time

def train(first_year, last_year, game_data='adv', window=2,
          train_size=0.8, learn_rate=0.00001, tol=0.002,
          n_learn_rate_changes=2, year_discount=0,
          verbose=True):

  if len(game_data) < 3:
    game_data = data_gather(first_year, last_year+1, game_data)#get rid of +1
  
  game_data, sos = data_init(game_data,first_year,last_year)

  nn_list = []
  for i in range(1,14):
    nn_list.append(NeuralNet((len(game_data.columns)-12)//2,
                         learn_rate, year_discount, tol))

  week = 13
  nn = nn_list[-1]
  for i in range(3):    
    train, test = custom_train_test_split(
      game_data, train_size, first_year, last_year, week-window, week+5)
    nn.epoch(train,last_year)
    nn.error_check(test,last_year)
    if verbose == True:
      print(week,'Train Error',round(nn.train_error,5),
          'Test Error',round(nn.test_error,5),
          round(nn.a,5),round(sum([abs(w) for W in nn.W1 for w in W]),5),
          nn.count)    

  ratings_calc(sos, nn, game_data)
  sos_calc(sos, game_data, first_year)

  i = 1
  while sum([nn.switch for nn in nn_list]) > 0:
    if verbose == True:
      print(i, sum([nn.switch for nn in nn_list]))
    for week in range(1,14):
      nn = nn_list[week-1]
      if nn.switch > 0:
        nn.n_worse += 1
        if week < 13:
          train, test = custom_train_test_split(
            game_data, train_size, first_year, last_year, week-window, week+window)
        else:
          train, test = custom_train_test_split(
            game_data, train_size, first_year, last_year, week-window, week+5)            
        
        nn.epoch(train, last_year)     
        nn.error_check(test, last_year)
        nn.assess(i, 4)
            
        if verbose == True:
          print(week,'Train Error:',round(nn.train_error,5),
                'Test Error:',round(nn.test_error,5),
                round(nn.a,5),round(sum([abs(w) for W in nn.W1 for w in W]),5),
                nn.count)
          
        if week == 13:
          for p in range(2):   
            ratings_calc(sos, nn, game_data)
            sos_calc(sos, game_data, first_year)
    i+=1

  for change in range(n_learn_rate_changes):
    if verbose == True:
      print('Learn Rate Change',change+1)

    for nn in nn_list:
      nn.switch = 1
      nn.n_worse = 0
      nn.learn *= 0.1

    j=1
    while sum([nn.switch for nn in nn_list]) > 0:
      if verbose == True:
        print(i, sum([nn.switch for nn in nn_list]))
      for week in range(1,14):
        nn = nn_list[week-1]
        if nn.switch > 0:
          nn.n_worse += 1
          if week < 13:
            train, test = custom_train_test_split(
              game_data, train_size, first_year, last_year, week-window, week+window)
          else:
            train, test = custom_train_test_split(
              game_data, train_size, first_year, last_year, week-window, week+5)            
          
          nn.epoch(train,last_year)     
          nn.error_check(test,last_year)
          nn.assess(j, 3)

          if verbose == True:
            print(week,'Train Error:',round(nn.train_error,5),
                  'Test Error:',round(nn.test_error,5),
                  round(nn.a,5),round(sum([abs(w) for W in nn.W1 for w in W]),5),
                  nn.count)

          if week == 13:
            for p in range(2):    
              ratings_calc(sos, nn, game_data)
              sos_calc(sos, game_data, first_year)
      i += 1
      j += 1

  return nn_list, sos, game_data
   
###################

game_data = pd.read_pickle('cfbdata\cfbfinalgamedata_temp.pkl')

nn_list,sos,game_data = train(first_year=2015,last_year=2018,game_data=game_data,
                          train_size=0.8,year_discount=0,tol=0.001,
                          learn_rate=0.0001,
                          n_learn_rate_changes=3,window=2)

##### TEST #####
loss2019 = 0
loss_tot = 0
num_right = 0
count = 0

nn = nn_list[0]
game_data20190 = game_data[
  (game_data['week'] < 2)&(game_data['season']==2019)
  ].reset_index(drop=True)

home20190 = game_data20190[
    game_data20190.columns[12:(len(game_data20190.columns)-12)//2 + 12]]
away20190 = game_data20190[
    game_data20190.columns[(len(game_data20190.columns)-12)//2 + 12:]]

week_loss = 0
week_count = 0

for i in range(len(game_data20190)):      
    
  X1 = home20190.iloc[i].values
  X2 = away20190.iloc[i].values
  
  s1 = nn.feedforward_ratingscalc(X1.astype('float32'))
  s2 = nn.feedforward_ratingscalc(X2.astype('float32'))

  #if math.isnan(s2):
  if game_data20190.loc[i,'away_conference'] == None:
    continue
    s2 = 0.2
  #elif math.isnan(s1):
  elif game_data20190.loc[i,'home_conference'] == None:
    continue
    s1 = 0.2
  
  neutral = game_data20190.loc[i, 'neutral']
  
  if neutral==0.0 or neutral==False:
    y_pred = nn.m*(s1-s2) + nn.a
  else:
    y_pred = nn.m*(s1-s2)

  if y_pred < 0 and game_data20190.loc[i, 'y_actual'] < 0:
    num_right += 1
  elif y_pred > 0 and game_data20190.loc[i, 'y_actual'] > 0:
    num_right += 1

  diff = game_data20190.loc[i, 'y_actual'] - y_pred
  loss_tot+=diff
  loss2019 += abs(diff)
  week_loss += abs(diff)
  week_count+=1
  count+=1

  home_team = game_data20190.loc[i,'home_team']
  away_team = game_data20190.loc[i,'away_team']
  week = game_data20190.loc[i,'week']

print(loss2019/count, count, loss_tot/count)

for week in range(2, 13):
  nn = nn_list[week-1]
  game_data20190 = game_data[
    (game_data['week'] == week)&(game_data['season']==2019)
    ].reset_index(drop=True)

  home20190 = game_data20190[
      game_data20190.columns[12:(len(game_data20190.columns)-12)//2 + 12]]
  away20190 = game_data20190[
      game_data20190.columns[(len(game_data20190.columns)-12)//2 + 12:]]

  week_loss = 0
  week_count = 0
  for i in range(len(game_data20190)):
      
    X1 = home20190.iloc[i].values
    X2 = away20190.iloc[i].values
    
    s1 = nn.feedforward_ratingscalc(X1.astype('float32'))
    s2 = nn.feedforward_ratingscalc(X2.astype('float32'))

    #if math.isnan(s2):
    if game_data20190.loc[i,'away_conference'] == None:
      continue
      s2 = 0.2
    #elif math.isnan(s1):
    elif game_data20190.loc[i,'home_conference'] == None:
      continue
      s1 = 0.2
      
    neutral = game_data20190.loc[i, 'neutral']
    
    if neutral==0.0 or neutral==False:
      y_pred = nn.m*(s1-s2) + nn.a
    else:
      y_pred = nn.m*(s1-s2)

    if y_pred < 0 and game_data20190.loc[i, 'y_actual'] < 0:
      num_right += 1
    elif y_pred > 0 and game_data20190.loc[i, 'y_actual'] > 0:
      num_right += 1

    diff = game_data20190.loc[i, 'y_actual'] - y_pred
    loss_tot += diff
    loss2019 += abs(diff)
    week_loss += abs(diff)
    week_count+=1
    count+=1

    home_team = game_data20190.loc[i,'home_team']
    away_team = game_data20190.loc[i,'away_team']
    week = game_data20190.loc[i,'week']

  print(week_loss/week_count, week_count, loss_tot/count)

nn = nn_list[-1]
game_data20190 = game_data[
  (game_data['week'] >= 13)&(game_data['week'] < 20)&
  (game_data['season']==2019)
  ].reset_index(drop=True)

home20190 = game_data20190[
    game_data20190.columns[12:(len(game_data20190.columns)-12)//2 + 12]]
away20190 = game_data20190[
    game_data20190.columns[(len(game_data20190.columns)-12)//2 + 12:]]

week_loss = 0
week_count = 0
for i in range(len(game_data20190)):
    
  X1 = home20190.iloc[i].values
  X2 = away20190.iloc[i].values
  
  s1 = nn.feedforward_ratingscalc(X1.astype('float32'))
  s2 = nn.feedforward_ratingscalc(X2.astype('float32'))

  #if math.isnan(s1):
  if game_data20190.loc[i,'away_conference'] == None:
    continue
    s2 = 0.2
  #elif math.isnan(s2):
  elif game_data20190.loc[i,'home_conference'] == None:
    continue
    s1 = 0.2
  
  neutral = game_data20190.loc[i, 'neutral']
  
  if neutral==0.0 or neutral==False:
    y_pred = nn.m*(s1-s2) + nn.a
  else:
    y_pred = nn.m*(s1-s2)

  if y_pred < 0 and game_data20190.loc[i, 'y_actual'] < 0:
    num_right += 1
  elif y_pred > 0 and game_data20190.loc[i, 'y_actual'] > 0:
    num_right += 1

  diff = game_data20190.loc[i, 'y_actual'] - y_pred
  loss_tot += diff
  loss2019 += abs(diff)
  week_loss += abs(diff)
  week_count+=1
  count+=1

  home_team = game_data20190.loc[i,'home_team']
  away_team = game_data20190.loc[i,'away_team']
  week = game_data20190.loc[i,'week']

  #preds2019[(home_team,away_team,week)]['pred'] += y_pred

print(week_loss/week_count, week_count, loss_tot/count)
print('Total Loss:',loss2019/count,count)
print('Number Correct:',num_right,num_right/count)

print(sos.sort_values('2019Rating',ascending=False).reset_index(
  drop=True).head(30).to_string())

df = pd.DataFrame()

j=0
for nn in nn_list:
  cols = []
  weights = []
  i=0
  sum0 = sum([abs(w) for W in nn.W1 for w in W])
  for col in game_data.columns[12:(len(game_data.columns)-12)//2 + 12]:
    avg = sum([abs(w) for w in nn.W1[i]])
    cols.append(col)
    weights.append(avg/sum0)
    i+=1      
  df['Stat'] = cols
  df['Weight'+str(j+1)] = weights
  j+=1

df = df.sort_values('Stat',ascending=False)
df['Weight_tot'] = sum([df[col] for col in df.columns[1:]])/13

print(df[['Stat','Weight_tot']].sort_values(
  'Weight_tot',ascending=False).reset_index(drop=True).to_string())
sys.exit()
