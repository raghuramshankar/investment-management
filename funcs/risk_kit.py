import pandas as pd
import numpy as np

def annualized_from_monthly(returns_m):
    total_returns_m = (returns_m + 1).prod() - 1
    returns_y = ((total_returns_m + 1)**(12/len(returns_m)) - 1) * 100
    volatility_y = (returns_m.std() * np.sqrt(12))* 100
    return returns_y, volatility_y

def max_drawdown(returns_m):
    wealth_index = 100*(1 + returns_m).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    drawdown_idxmin = drawdown.idxmin()
    drawdown_min = drawdown.min() * 100
    return drawdown_idxmin, drawdown_min