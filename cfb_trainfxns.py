import numpy as np
import pandas as pd
import copy
    
class NeuralNet():
  def __init__(self, 
               n, 
               learn_rate = 0.0001, 
               year_discount = 0,
               tol = 0.001):
    self.learn = learn_rate
    self.year_discount = year_discount
    self.tol = tol
    
    self.W1 = np.random.normal(scale=0.00001, size=[n,n//2]).astype('float32')
    self.W2 = np.random.normal(scale=0.00001, size=[n//2,]).astype('float32')
    self.b1 = np.random.normal(scale=0.0001, size=[n//2,]).astype('float32')
    self.b2 = np.random.normal(scale=0.0001)
    self.m = 100
    self.a = 2.0

    self.train_error = 0.0
    self.test_error = 0.0
    self.best_test_error = 1000
    self.n_worse = 0
    self.switch = 1

  def feedforward_train(self, X1):
    F1 = sigmoid(np.dot(self.W1.T, X1) + self.b1)
    return sigmoid(np.dot(self.W2, F1) + self.b2), F1

  def feedforward_ratingscalc(self, X1):
    return sigmoid(
      np.dot(self.W2, sigmoid(np.dot(self.W1.T, X1) + self.b1)) + self.b2)

  def margin_predict(self, s1, s2, neutral):
    if neutral == False:
      return self.m*(s1-s2) + self.a            
    else:
      return self.m*(s1-s2)
    
  def epoch(self, train, last_year):
    n_cols = len(train.columns) - 12
    self.total_loss = 0
    self.count = 0
    np.apply_along_axis(self.update,1,train,last_year,n_cols)
    self.train_error = self.total_loss/self.count

  def update(self, game, last_year, n_cols):    
    if game[3] == None:
      return
    if game[2] == None:
      return
    X2 = game[n_cols//2 + 12:].astype('float32')
    s2,F2 = self.feedforward_train(X2)

    ds2dm2 = s2*(1-s2)                
    dm2dW2 = F2
    ds2dW2 = ds2dm2*dm2dW2
    ds2db2 = ds2dm2

    dm2dF2 = self.W2
    dF2dG2 = F2*(1-F2)
    dm2dG2 = dm2dF2*dF2dG2
    dG2dW1 = X2.reshape((n_cols)//2,1)
    dm2dW1 = np.dot(dG2dW1, dm2dG2.reshape(1, n_cols//4))
    ds2dW1 = ds2dm2*dm2dW1
    dm2db1 = dm2dG2
    ds2db1 = ds2dm2*dm2db1

    X1 = game[12:n_cols//2 + 12].astype('float32')
    s1,F1 = self.feedforward_train(X1)

    ds1dm1 = s1*(1-s1)                
    dm1dW2 = F1
    ds1dW2 = ds1dm1*dm1dW2
    ds1db2 = ds1dm1

    dm1dF1 = self.W2
    dF1dG1 = F1*(1-F1)
    dm1dG1 = dm1dF1*dF1dG1
    dG1dW1 = X1.reshape(n_cols//2,1)
    dm1dW1 = np.dot(dG1dW1, dm1dG1.reshape(1,n_cols//4))
    ds1dW1 = ds1dm1*dm1dW1
    dm1db1 = dm1dG1
    ds1db1 = ds1dm1*dm1db1

    neutral = game[6]
    y_pred = self.margin_predict(s1, s2, neutral)
    
    season = game[4]
    r = 1 - self.year_discount*(last_year-season)    
    diff = game[7] - y_pred
    self.total_loss += r*abs(diff)
    self.count += r

    dLdy_pred = -2*diff
    
    dy_preddW2 = self.m*(ds1dW2 - ds2dW2)
    dLdW2 = dLdy_pred*dy_preddW2
    
    dy_preddb2 = self.m*(ds1db2 - ds2db2)
    dLdb2 = dLdy_pred*dy_preddb2
    
    dy_preddW1 = self.m*(ds1dW1 - ds2dW1)
    dLdW1 = dLdy_pred*dy_preddW1
    
    dy_preddb1 = self.m*(ds1db1 - ds2db1)
    dLdb1 = dLdy_pred*dy_preddb1
    
    if neutral == False:
      self.a -= r*self.learn*dLdy_pred

    self.W2 -= r*self.learn*dLdW2
    self.b2 -= r*self.learn*dLdb2
    self.W1 -= r*self.learn*dLdW1
    self.b1 -= r*self.learn*dLdb1

  def error_check(self, test, last_year):
    n_cols = len(test.columns) - 12
    test_loss = 0
    count = 0
    for i in range(len(test)):      
      if test.loc[i, 'away_conference'] == None:
        continue
      elif test.loc[i, 'home_conference'] == None:
        continue

      X1 = test.loc[i, test.columns[12:n_cols//2 + 12]].astype('float32')
      X2 = test.loc[i, test.columns[n_cols//2 + 12:]].astype('float32')
      
      s1 = self.feedforward_ratingscalc(X1)
      s2 = self.feedforward_ratingscalc(X2)
      
      neutral = test.loc[i, 'neutral']      
      y_pred = self.margin_predict(s1, s2, neutral)

      season = test.loc[i, 'season']
      r = 1 - self.year_discount*(last_year-season)
      test_loss += r*abs(test.loc[i, 'y_actual'] - y_pred)
      count += r

    self.test_error = test_loss/count

  def assess(self, counter, threshold):
    if self.test_error > (self.best_test_error - self.tol) and \
       self.n_worse >= 2 and counter >= threshold:
      self.switch = 0
      self.W1 = copy.deepcopy(self.W1_best)
      self.W2 = copy.deepcopy(self.W2_best)
      self.b1 = copy.deepcopy(self.b1_best)
      self.b2 = copy.deepcopy(self.b2_best)
      self.a = copy.deepcopy(self.a_best)
    elif self.test_error < self.best_test_error:
      if self.test_error <= (self.best_test_error - self.tol):
        self.n_worse = 0
      self.best_test_error = self.test_error
      self.W1_best = copy.deepcopy(self.W1)
      self.W2_best = copy.deepcopy(self.W2)
      self.b1_best = copy.deepcopy(self.b1)
      self.b2_best = copy.deepcopy(self.b2)
      self.a_best = copy.deepcopy(self.a)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def ratings_calc(sos, nn, game_data):
  game_data1 = game_data[game_data.week == 20].reset_index(drop=True)
  col_cutoff = (len(game_data1.columns)-12)//2 + 12
  np.apply_along_axis(team_rating,1,game_data1,sos,game_data,nn,col_cutoff)

def team_rating(tg, sos, game_data, nn, col_cutoff):
  team = tg[0]
  season = tg[4]

  s1 = nn.feedforward_ratingscalc(tg[12:col_cutoff].astype('float32'))  
  sos.loc[sos.Team == team, str(season)+'Rating'] = s1

  if season != 2019: 
    game_data.loc[((game_data.home_team == team)&
                   (game_data.season == season+1)),
                  'home_last_rating'] = s1
    game_data.loc[((game_data.away_team == team)&
                   (game_data.season == season+1)),
                  'away_last_rating'] = s1

def team_sos(team_info,sos,game_data,first_year):
  team = team_info[0]
  team_games = game_data[(game_data.home_team == team)|
                         (game_data.away_team == team)]
  season=first_year
  for info in team_info[1:len(team_info)//2+1]:
    if info!='FCS':
      tg = team_games[team_games.season == season].sort_values('week')
      opps = tg.iloc[-1].home_opponents[:-1]
      a = np.array([sos.loc[sos.Team == opp,
                            str(season)+'Rating'].values[0] for opp in opps])
      b=pd.Series(np.concatenate([[0.5],np.cumsum(a)/np.arange(1,len(a)+1)]),
              tg.index,
              dtype='float32')
      
      tg_home_index = tg[tg.home_team==team].index
      tg_away_index = tg[tg.away_team==team].index
      game_data.loc[tg_home_index,'home_SOS'] = b.loc[tg_home_index]
      game_data.loc[tg_away_index,'away_SOS'] = b.loc[tg_away_index]
      
    season+=1

def sos_calc(sos,game_data,first_year):
  np.apply_along_axis(team_sos,1,sos,sos,game_data,first_year)
