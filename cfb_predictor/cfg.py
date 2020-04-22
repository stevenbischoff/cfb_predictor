def init(week_discount, fs, ls):
  global learn_rate_counter, n_cols, threshold, total_rounds, r, ratings_dict, index_dict, nn_list
  global first_season, last_season
  learn_rate_counter = 1
  threshold = 6
  total_rounds = 1
  r = 1 - week_discount
  index_dict = {}
  ratings_dict = {}
  nn_list = []
  first_season = fs
  last_season = ls
