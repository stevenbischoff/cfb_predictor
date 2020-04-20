import numpy as np
import pandas as pd
import copy
import data_fxns
import cfg

class NeuralNet():
  def __init__(self, week, window = 2, learn_rate = 0.0001, season_discount = 0, tol = 0.0001):     
    """
    Initializes a neural network
    ----------
    Parameters
    ----------
      week, window: int
      learn_rate: float
      season_discount: int or float
        If non-zero, discounts less recent seasons during training and error calculation
        A season's contribution to the total loss function is multiplied by (1 - season_discount*(last_season - season)).
        For instance, if season_discount = 0.05 and the last season is 2019, the loss from 2016 is discounted by 15%.
      tol: float
        The tolerance for early stopping. Training (at the current learning rate) stops once a fixed number of training rounds
        has been reached and the test error fails to improve by at least tol over two consecutive rounds.
    ----------
    Attributes
    ----------
      self.m, self.a
        Predicted spread is self.m * (home_rating - away_rating) + self.a
        self.a is the home-field advantage and is trained.
      self.best_mean_test_error
        Stores the best test set error so far during training
      self.n_worse
        How many consecutive rounds the test error has failed to improve by tol
      self.switch
        If set to 0, training for self is paused or finished
      self.week
        The week that the Neural Net is trained to predict
      self.window
        The window of weeks around self.week that the Neural Net is trained on
    """    
    self.learn = learn_rate
    self.season_discount = season_discount
    self.tol = tol
    self.week = week
    self.window = window
    
    self.W1 = np.random.normal(scale=0.00001, size=[cfg.n_cols,cfg.n_cols//2]).astype('float32')
    self.W2 = np.random.normal(scale=0.00001, size=[cfg.n_cols//2,]).astype('float32')
    self.b1 = np.random.normal(scale=0.0001, size=[cfg.n_cols//2,]).astype('float32')
    self.b2 = np.random.normal(scale=0.0001)
    self.m = 80
    self.a = 2.0

    self.mean_train_error = None
    self.mean_test_error = None
    self.best_mean_test_error = 1000

    self.n_worse = 0
    self.switch = 1


  def __str__(self):
    return 'Week: {}'.format(self.week) + ' Train Error: {}'.format(round(self.mean_train_error, 5)
      ) + ' Test Error: {}'.format(round(self.mean_test_error, 5))


  def reset(self):
    """
    Prepares self for training at a new learning rate
    """
    self.n_worse = 0
    self.switch = 1    
    self.learn *= 0.2


  def init_train_test(self, game_data, train_size):
    """
    Splits the games in self's window into train and test sets
    ----------
    Parameters
    ----------
      game_data: DataFrame
      train_size: float
    """
    if self.week < 13:
      self.train, self.test = data_fxns.custom_train_test_split(
        game_data, train_size, self.week-self.window, self.week+self.window)
    else:
      self.train, self.test = data_fxns.custom_train_test_split(
        game_data, train_size, self.week-self.window, self.week+6)


  def sigmoid(self, X):
    return 1 / (1 + np.exp(-X))

   
  def feedforward_train(self, X):
    """
    Returns a team's rating and the hidden layer array, which is required for backpropagation
    ----------
    Parameters
    ----------
      X: array
    """
    F1 = self.sigmoid(np.dot(self.W1.T, X) + self.b1)
    return self.sigmoid(np.dot(self.W2, F1) + self.b2), F1


  def feedforward(self, X):
    """
    Rating calculation for use when not doing backpropagation
    ----------
    Parameters
    ----------
      X: array
    """
    return self.sigmoid(np.dot(self.W2, self.sigmoid(np.dot(self.W1.T, X) + self.b1)) + self.b2)


  def margin_predict(self, s1, s2, neutral):
    """
    Returns a predicted spread. The prediction adds a home field advantage iff the game is not played on a neutral field.
    ----------
    Parameters
    ----------
      s1, s2: float
        Home rating, away rating
      neutral: bool
        True iff the game is played on a neutral field
    """
    if neutral == False:
      return self.m*(s1 - s2) + self.a            
    else:
      return self.m*(s1 - s2)
 
    
  def epoch(self, train):  
    """
    A single epoch of stochastic gradient descent applied to the training data. Calculates the training error for that epoch.
    ----------
    Parameters
    ----------
      train: DataFrame
    """
    self.total_train_error = 0
    self.count = 0
    np.apply_along_axis(self.update, 1, train)
    self.mean_train_error = self.total_train_error/self.count

    
  def update(self, game):
    """
    Prediction and backpropagation for a single data point.
    ----------
    Parameters
    ----------
      game: array
    -----
    Notes
    -----
     - The first four lines skip the data point iff one of the teams is not an FBS team, i.e. not in the highest level of NCAA football.
       The collegefootballdata API lists a team's conference as None iff the team does not play at the FBS level.
    """
    if game[4] == None: # game[4] is the away_conference
      return
    if game[3] == None: # game[3] is the home_conference
      return

    X1 = game[13:cfg.n_cols + 13].astype('float32')
    s1, F1 = self.feedforward_train(X1)

    ds1dm1 = s1*(1 - s1)
    ds1dW2 = ds1dm1*F1
    ds1db2 = ds1dm1

    dm1dG1 = self.W2*F1*(1 - F1)
    ds1dW1 = ds1dm1*np.dot(X1.reshape(cfg.n_cols, 1), dm1dG1.reshape(1, cfg.n_cols//2))
    ds1db1 = ds1dm1*dm1dG1

    X2 = game[cfg.n_cols + 13:].astype('float32')
    s2, F2 = self.feedforward_train(X2)

    ds2dm2 = s2*(1 - s2)
    ds2dW2 = ds2dm2*F2
    ds2db2 = ds2dm2

    dm2dG2 = self.W2*F2*(1 - F2)
    ds2dW1 = ds2dm2*np.dot(X2.reshape(cfg.n_cols, 1), dm2dG2.reshape(1, cfg.n_cols//2))
    ds2db1 = ds2dm2*dm2dG2

    neutral = game[7]     
    game_error = game[8] - self.margin_predict(s1, s2, neutral) # game[8] is the game's actual spread

    r = 1 - self.season_discount*(cfg.last_season - game[5]) # Calculate season weight. game[5] is the game's season  
    self.total_train_error += r*abs(game_error)
    self.count += r

    dLdy_pred = -2*r*game_error 
    dLds1s2 = dLdy_pred*self.m
    
    if neutral == False:
      self.a -= self.learn*dLdy_pred
    self.W2 -= self.learn*dLds1s2*(ds1dW2 - ds2dW2)
    self.b2 -= self.learn*dLds1s2*(ds1db2 - ds2db2)
    self.W1 -= self.learn*dLds1s2*(ds1dW1 - ds2dW1)
    self.b1 -= self.learn*dLds1s2*(ds1db1 - ds2db1)

    
  def error_check(self, test):
    """
    Calculates the total error of a set of datapoints without performing backpropagation
    ----------
    Parameters
    ----------
      test: DataFrame
    """
    self.total_test_error = 0
    self.count = 0
    np.apply_along_axis(self.game_error, 1, test)
    self.mean_test_error = self.total_test_error/self.count
  

  def game_error(self, game):
    """
    Same as self.update without backpropagation
    """
    if game[4] == None:
      return
    elif game[3] == None:
      return
    
    X1 = game[13:cfg.n_cols + 13].astype('float32')
    X2 = game[cfg.n_cols + 13:].astype('float32')
    
    s1 = self.feedforward(X1)
    s2 = self.feedforward(X2)
     
    y_pred = self.margin_predict(s1, s2, game[7])

    r = 1 - self.season_discount*(cfg.last_season - game[5])
    self.total_test_error += r*abs(game[8] - y_pred)
    self.count += r

 
  def assess(self, learn_rate_counter, threshold): 
    """
    Assesses whether self should continue training
    ----------
    Parameters
    ----------
      learn_rate_counter, threshold: int
    -----
    Notes
    -----
      If self.mean_test_error has not improved by at least tol over 2 consecutive training rounds and a minimum threshold for number
      of training rounds has been met, set self.switch equal to 0 and restore the best version of the network.
    
      If self.mean_test_error is below the previous best mean_test_error, reset the number of round without improvement to 0,
      store the new best error, and store copies of the current parameters as the current best version of the network
    """
    if self.mean_test_error > (self.best_mean_test_error - self.tol) and self.n_worse >= 2 and\
       cfg.learn_rate_counter >= cfg.threshold:
      self.switch = 0
      self.W1 = copy.deepcopy(self.W1_best)
      self.W2 = copy.deepcopy(self.W2_best)
      self.b1 = copy.deepcopy(self.b1_best)
      self.b2 = copy.deepcopy(self.b2_best)
      self.a = copy.deepcopy(self.a_best)
    elif self.mean_test_error < self.best_mean_test_error:
      if self.mean_test_error <= (self.best_mean_test_error - self.tol):
        self.n_worse = 0
      self.best_mean_test_error = self.mean_test_error
      self.W1_best = copy.deepcopy(self.W1)
      self.W2_best = copy.deepcopy(self.W2)
      self.b1_best = copy.deepcopy(self.b1)
      self.b2_best = copy.deepcopy(self.b2)
      self.a_best = copy.deepcopy(self.a)
