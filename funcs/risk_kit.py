import pandas as pd
import numpy as np
import scipy.stats

def read_dataframe(filename, format):
    df = pd.read_csv('data/' + filename, header=0, index_col=0)
    df.index = pd.to_datetime(pd.to_datetime(df.index, format=format))
    df.columns = df.columns.str.strip()
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
    return drawdown, drawdown_idxmin, drawdown_min

def jb_test(returns_m, level=0.01):
    _, p_value = scipy.stats.jarque_bera(returns_m)
    return p_value > level

def skew(returns_m):
    skewness = pd.Series(scipy.stats.skew(returns_m), index=returns_m.columns)
    return skewness

def kurtosis(returns_m):
    kurtosis = pd.Series(scipy.stats.kurtosis(returns_m), index=returns_m.columns)
    return kurtosis

def negative_semideviation(returns_m):
    return returns_m[returns_m>0].std(ddof=0)

def historic_var(returns_m, level=5):
    return -(returns_m.aggregate(np.percentile, q=level, axis=0))

def gaussian_var(returns_m, level=5, modified=False):
    z = scipy.stats.norm.ppf(level/100)
    if modified == True:
        s = skew(returns_m)
        k = kurtosis(returns_m)
        z = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z*3 - 5*z)*(s**2)/36
    return -(returns_m.mean() + z * returns_m.std(ddof=0))

def historic_cvar(returns_m, level=5):
    return - returns_m[returns_m <= -historic_var(returns_m, level)].mean()