import pandas as pd
import numpy as np
import scipy.stats

def read_dataframe(filename, format):
    df = pd.read_csv('data/' + filename, header=0, index_col=0)
    df.index = pd.to_datetime(pd.to_datetime(df.index, format=format))
    df.columns = df.columns.str.strip()
    return df

def annualized_returns(returns, period):
    total_returns = (returns + 1).prod() - 1
    returns_y = ((total_returns + 1)**(period/len(returns)) - 1) * 100
    volatility_y = (returns.std() * np.sqrt(period))* 100
    return returns_y, volatility_y

def max_drawdown(returns):
    wealth_index = 100*(1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    drawdown_idxmin = drawdown.idxmin()
    drawdown_min = drawdown.min() * 100
    return drawdown, drawdown_idxmin, drawdown_min

def jb_test(returns, level=0.01):
    _, p_value = scipy.stats.jarque_bera(returns)
    return p_value > level

def skew(returns):
    skewness = pd.Series(scipy.stats.skew(returns), index=returns.columns)
    return skewness

def kurtosis(returns):
    kurtosis = pd.Series(scipy.stats.kurtosis(returns), index=returns.columns)
    return kurtosis

def negative_semideviation(returns):
    return returns[returns>0].std(ddof=0)

def historic_var(returns, level=5):
    return -(returns.aggregate(np.percentile, q=level, axis=0))

def gaussian_var(returns, level=5, modified=False):
    z = scipy.stats.norm.ppf(level/100)
    if modified == True:
        s = skew(returns)
        k = kurtosis(returns)
        z = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2*z*3 - 5*z)*(s**2)/36
    return -(returns.mean() + z * returns.std(ddof=0))

def historic_cvar(returns, level=5):
    return - returns[returns <= -historic_var(returns, level)].mean()

def sharpe_ratio(returns, period, risk_free_rate_m=3):
    riskfree_returns = (1 + risk_free_rate_m/100)**(1/12) - 1
    excess_returns = returns - riskfree_returns
    excess_returns_y, _ = annualized_returns(excess_returns, period)
    _, volatility_y = annualized_returns(returns, period)
    return excess_returns_y/volatility_y