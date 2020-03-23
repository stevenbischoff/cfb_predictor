# cfb_predictor
Rating college football teams

## Rating system
The model rates college football teams using neural networks with one hidden layer. The network has been built from scratch in Python, using only pandas, numpy, and Python's built-in packages.

### Input data
For a given team _t_ and week of the season _w_, the model calculates a rating for _t_ at _w_ using _t_'s cumulative statistics for all weeks _w0_ < _w_. The package includes modules to pull and appropriately transform statistics from the collegefootballapi located here: https://api.collegefootballdata.com/api/docs/?url=/api-docs.json#/. Right now, the package is only able to pull "advanced" per-game statistics.

In addition to the per-game statistics, the model's rating calculations also depend on a team's talent, their strength of schedule (SOS), and last season's rating. Talent ratings are 247's composite talent ratings pulled from the same source. These ratings only go back to 2015. 

## Training
The model is trained by predicting final spreads (home team points - away team points). To make a prediction, the model calculates a rating (between 0 and 1) for the home and away teams. The predicted spread is 100*(home team rating - away team rating) + _a_, where _a_ represents home-field advantage and is trained simultaneously with the ratings neural networks. Training uses backpropagation, specifically stochastic gradient descent.  
