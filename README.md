# cfb_predictor
Rating college football teams

## Installation

## Rating system
The model rates college football teams using neural networks with one hidden layer. The model has been built from scratch in Python, using only pandas, numpy, and Python's built-in packages.

### Input data
For a given team _t_ and week of the season _w_, the model calculates a rating for _t_ at _w_ using _t_'s cumulative statistics for all weeks _w0_ < _w_. The package includes modules to pull and appropriately transform statistics from the collegefootballapi located here: https://api.collegefootballdata.com/api/docs/?url=/api-docs.json#/. Right now, the package is only able to pull "advanced" per-game statistics.

In addition to the per-game statistics, the model's rating calculations also depend on a team's talent, their strength of schedule (SOS), and last season's rating. Talent ratings are 247's composite talent ratings pulled from the same source. These ratings only go back to 2015.

### Different weeks of the season, different neural networks
As the season progresses, the importance of different types of inputs for prediction can change drastically. For instance, early in the season a team's last rating and talent will be much more predictive of their performance than their cumulative efficiency statistics, since the efficiency statistics will be based on few or no games. As more games are played, the larger sample will make these efficiency statistics much more predictive of future performance.

With this in mind, the model actually uses multiple neural networks, one each for weeks 1-13+. Though this comes with a cost of a smaller training set for the individual neural networks, I've found that this does lead to better peformance than a single neural network for every week. 

## Training
The model is trained by predicting final spreads (home team points - away team points). To make a prediction, the model calculates a rating (between 0 and 1) for the home and away teams. The predicted spread is 100*(home team rating - away team rating) + _a_, where _a_ represents home-field advantage and is trained simultaneously with the ratings neural networks. Training uses backpropagation, specifically stochastic gradient descent with early stopping.  

The neural network for week _w_ is trained using data from an adjustable window around _w_.

## Results
Trained on the 2045-2018 seasons, the model regularly achieve a mean absolute error of < 13 points when predicting the spread of FBS games in the 2019 season. For comparison with other rating systems, follow this link: http://www.thepredictiontracker.com/ncaaresults.php?orderby=absdev&type=1&year=19 . 

The below tables display the model's best and worst predictions:

![Best Predictions](best.png)

![Worst Predictions](worst.png)
