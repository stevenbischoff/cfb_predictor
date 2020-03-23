# cfb_predictor
Rating college football teams

## Rating system
The model rates college football teams using neural networks with one hidden layer. The network has been built from scratch in Python, using only pandas, numpy, and Python's built-in packages.

### Input data
For a given team _t_ and week of the season _w_, the model calculates a rating for _t_ at _w_ using _t_'s cumulative statistics for all weeks _w0_ < _w_. The package includes modules to pull and appropriately transform statistics from the collegefootballapi located here: https://api.collegefootballdata.com/api/docs/?url=/api-docs.json#/. Right now, the package is only able to pull "advanced" per-game statistics.

### Different networks for different weeks 
