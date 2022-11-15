import pandas as pd
import numpy as np
import scipy.stats

def read_dataframe(filename, format):
    df = pd.read_csv('../data/' + filename, header=0, index_col=0)
    df.index = pd.to_datetime(pd.to_datetime(df.index, format=format))
    return df

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

def jb_test(returns_m, level=0.01):
    _, p_value = scipy.stats.jarque_bera(returns_m)
    return p_value > level

def skew(returns_m):
    skewness = pd.Series(scipy.stats.skew(returns_m), index=returns_m.columns)
    return skewness

def kurtosis(returns_m):
    kurtosis = pd.Series(scipy.stats.kurtosis(returns_m), index=returns_m.columns)
    return kurtosis