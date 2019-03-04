# A lightweight tear sheet generator 

_Want some nice stats for your investment strategies, but don't want to mess with the gigantic pyfolio package?_

**You're in luck!**

This framework borrows from `pyfolio` but is drastically minimized to avoid bloat. No dependencies except for `numpy` and `pandas` (and matplotlib if you want the plots). 

There's probably some more statistics you might want - please open an issue and tell me how ignorant I am of the One Metric to Rule Them All and I will likely listen to you and learn something - these are just the ones I use.

## How to use:

1) Prepare your strategy for evaluation by creating a `pd.Series` of the daily returns of your strategy, and the daily returns of a benchmark strategy.

2) Feed the returns to `create_tear_sheet` function. 

3) Voila! 

I've included sample strategy data (buy-n-hold AAPL as your strategy, buy-n-hold SPY as your benchmark) and a notebook for you to play with and see how it all works. 

Have fun! And if you're just getting started with algotrading:

> First, ask youself if the allure of skimming a few pennies off the top of a corrupt system that will overtake your life, blacken your soul, and bankrupt you in the process is the best use of your talents. No? Good choice; back to Hacker News. Yes? Get a second opinion from your mother. I’ll wait.
~ "Doctor J"
