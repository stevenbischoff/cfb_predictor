import numpy as np
import pandas as pd
import copy
    
class NeuralNet():
  def __init__(self, n, week, learn_rate = 0.0001, season_discount = 0, tol = 0.0001): 
    
    """
    Initializes a neural network
    ----------
    Parameters
    ----------
      n: int
        The number of inputs
      learn_rate: float
      season_discount: int or float
        If non-zero, discounts less recent seasons during training and error calculation
        A season's contribution to the total loss function is multiplied by season_discount * (last_season - season).
        So if season_discount = 0.05 and the last season is 2019, the loss from 2016 is discounted by 15%.
      tol: float
        The tolerance for early stopping. Training (at the current learning rate) stops once a fixed number of training rounds
        has been reached and the test error fails to improve by at least tol over two consecutive rounds.
    ----------
    Attributes
    ----------
      self.m, self.a
        Predicted spread is self.m * (home_rating - away_rating) + self.a
        self.a is trained.
      self.best_test_error
        Stores the best test set error so far during training
      self.n_worse
        How many consecutive rounds the test error has failed to improve by tol
      self.switch
        If set to 0, training for self is paused or finished
      self.week
        The week that the Neural Net is trained to predict
    """
    
    self.learn = learn_rate
    self.season_discount = season_discount
    self.tol = tol

    self.week = week
    
    self.W1 = np.random.normal(scale=0.00001, size=[n,n//2]).astype('float32')
    self.W2 = np.random.normal(scale=0.00001, size=[n//2,]).astype('float32')
    self.b1 = np.random.normal(scale=0.0001, size=[n//2,]).astype('float32')
    self.b2 = np.random.normal(scale=0.0001)
    self.m = 80
    self.a = 2.0

    self.train_error = 0.0
    self.test_error = 0.0
    self.best_test_error = 1000
    self.n_worse = 0
    self.switch = 1

    
  def feedforward_train(self, X):
    """
    Returns a team's rating and the hidden layer array, which is required for backpropagation
    ----------
    Parameters
    ----------
      X1: array
    """
    F1 = sigmoid(np.dot(self.W1.T, X) + self.b1)
    return sigmoid(np.dot(self.W2, F1) + self.b2), F1


  def feedforward_ratingscalc(self, X1):
    """
    Rating calculation for use when not doing backpropagation
    """
    return sigmoid(np.dot(self.W2, sigmoid(np.dot(self.W1.T, X1) + self.b1)) + self.b2)


  def margin_predict(self, s1, s2, neutral):
    """
    Returns a predicted spread. The prediction adds a home field advantage iff the game is not played on a neutral field.
    ----------
    Parameters
    ----------
      s1, s2: float
        Home rating, away rating
      neutral: bool
        False iff the game is not played on a neutral field
    """
    if neutral == False:
      return self.m*(s1 - s2) + self.a            
    else:
      return self.m*(s1 - s2)
 
    
  def epoch(self, train, last_season):  
    """
    A single epoch of stochastic gradient descent applied to the training data. Calculates the training error for that epoch.
    ----------
    Parameters
    ----------
      train: DataFrame
      last_season: int
    """
    n_cols = (len(train.columns) - 12)//2
    self.total_loss = 0
    self.count = 0
    np.apply_along_axis(self.update, 1, train, last_season, n_cols)
    self.train_error = self.total_loss/self.count

    
  def update(self, game, last_season, n_cols):
    """
    Prediction and backpropagation for a single data point.
    ----------
    Parameters
    ----------
      game: DataFrame
      last_season: int
      n_cols: int
    -----
    Notes
    -----
     - The first four lines skip the data point iff one of the teams is not an FBS team, i.e. not in the highest level of NCAA football.
       game[2] and game[3] are the conferences of the home team and away team. The collegefootballdata API lists a team's conference
       as None iff the team does not play at the FBS level.
     - During backpropagation, so many intermediate values are stored mostly to help me keep track of things.
    """
    if game[3] == None:
      return
    if game[2] == None:
      return

    X2 = game[n_cols + 12:].astype('float32')
    s2, F2 = self.feedforward_train(X2)

    ds2dm2 = s2*(1 - s2)                
    dm2dW2 = F2
    ds2dW2 = ds2dm2*dm2dW2
    ds2db2 = ds2dm2

    dm2dF2 = self.W2
    dF2dG2 = F2*(1 - F2)
    dm2dG2 = dm2dF2*dF2dG2
    dG2dW1 = X2.reshape(n_cols, 1)
    dm2dW1 = np.dot(dG2dW1, dm2dG2.reshape(1, n_cols//2))
    ds2dW1 = ds2dm2*dm2dW1
    dm2db1 = dm2dG2
    ds2db1 = ds2dm2*dm2db1

    X1 = game[12:n_cols + 12].astype('float32')
    s1,F1 = self.feedforward_train(X1)

    ds1dm1 = s1*(1 - s1)                
    dm1dW2 = F1
    ds1dW2 = ds1dm1*dm1dW2
    ds1db2 = ds1dm1

    dm1dF1 = self.W2
    dF1dG1 = F1*(1 - F1)
    dm1dG1 = dm1dF1*dF1dG1
    dG1dW1 = X1.reshape(n_cols, 1)
    dm1dW1 = np.dot(dG1dW1, dm1dG1.reshape(1, n_cols//2))
    ds1dW1 = ds1dm1*dm1dW1
    dm1db1 = dm1dG1
    ds1db1 = ds1dm1*dm1db1

    neutral = game[6]
    y_pred = self.margin_predict(s1, s2, neutral)
    
    season = game[4]
    r = 1 - self.season_discount*(last_season - season)    
    game_error = game[7] - y_pred
    self.total_loss += r*abs(game_error)
    self.count += r

    dLdy_pred = -2*game_error
    
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

    
  def error_check(self, test, last_season):
    """
    Calculates the total error of a set of datapoints without performing backpropagation
    ----------
    Parameters
    ----------
      test: DataFrame
      last_season: int
    """
    n_cols = (len(test.columns) - 12)//2
    self.test_loss = 0
    self.count = 0
    np.apply_along_axis(self.game_error, 1, test, last_season, n_cols)
    self.test_error = self.test_loss/self.count
  

  def game_error(self, game, last_season, n_cols):
    """
    Same as self.update without backpropagation
    """
    if game[3] == None:
      return
    elif game[2] == None:
      return
    
    X1 = game[12:n_cols + 12].astype('float32')
    X2 = game[n_cols + 12:].astype('float32')
    
    s1 = self.feedforward_ratingscalc(X1)
    s2 = self.feedforward_ratingscalc(X2)

    neutral = game[6]      
    y_pred = self.margin_predict(s1, s2, neutral)

    season = game[4]
    r = 1 - self.season_discount*(last_season - season)
    self.test_loss += r*abs(game[7] - y_pred)
    self.count += r
    
 
  def assess(self, counter, threshold): 
    """
    If self.test_error has not improved by at least tol over 2 consecutive training rounds and a minimum threshold for number of
    of training rounds has been met, set self.switch equal to 0 and restore the best-peforming parameters.
    If self.test_error is below the previous best test_error, resets the number of round without improvement to 0,
    stores the new best error, and stores copies of the current parameters as the current best version of the network
    """
    if self.test_error > (self.best_test_error - self.tol) and self.n_worse >= 2 and counter >= threshold:
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


def ratings_calc(ratings_dict, nn, game_data):
  """
  Calculates the final ratings for every FBS team and updates the last season ratings in game_data 
  Each team's final statistics are stored in a "week 20" row of game_data
  ----------
  Parameters
  ----------
    sos: DataFrame
    nn: NeuralNet
    game_data: DataFrame
  """
  last_season = max(game_data.season)
  game_data1 = game_data[game_data.week == 20]
  col_cutoff = (len(game_data1.columns)-12)//2 + 12
  np.apply_along_axis(team_rating, 1, game_data1, ratings_dict, game_data, nn, col_cutoff, last_season)


def team_rating(tg, ratings_dict, game_data, nn, col_cutoff, last_season):
  """
  Calculates a team's final rating for a particular season. Updates the SOS dataframe with this information, and also updates
  the last_rating information in game_data iff the season is less than last_season
  ----------
  Parameters
  ----------
    tg: Series
    sos, game_data: DataFrame
    nn: NeuralNet
    col_cutoff, last_season: int
  """
  team = tg[0]
  season = tg[4]

  s1 = nn.feedforward_ratingscalc(tg[12:col_cutoff].astype('float32'))  
  ratings_dict[team][str(season)+'Rating'] = s1
  if season < last_season:
    game_data.at[((game_data.home_team == team)&(game_data.season == season + 1)),
                  'home_last_rating'] = s1
    game_data.at[((game_data.away_team == team)&(game_data.season == season + 1)),
                  'away_last_rating'] = s1


def sos_calc(ratings_dict, game_data, first_season, last_season):
  """
  For each team, performs Strength of Schedule (sos_dict) calculations for each season for which they were in the FBS
  This function takes up a majority of training time, and I'm always looking for ways to improve its speed.
  """
  for team in ratings_dict.keys():
    team_games = game_data[(game_data.home_team == team)|(game_data.away_team == team)]
    for season in range(first_season, last_season + 1):
      if ratings_dict[team][str(season)+'League'] != 'FCS':
        tg = team_games[team_games.season == season].sort_values('week')
        opps = tg.iloc[-1].home_opponents[:-1]
        a = np.array([ratings_dict[opp][str(season) + 'Rating'] for opp in opps]) 
        avg_sos = pd.Series(np.concatenate([[0.5], np.cumsum(a)/np.arange(1, len(a) + 1)]), tg.index)
        
        tg_home_index = tg[tg.home_team==team].index
        tg_away_index = tg[tg.away_team==team].index
        game_data.at[tg_home_index,'home_SOS'] = avg_sos.loc[tg_home_index]
        game_data.at[tg_away_index,'away_SOS'] = avg_sos.loc[tg_away_index]
