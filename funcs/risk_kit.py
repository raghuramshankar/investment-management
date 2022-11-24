import pandas as pd
import numpy as np
import scipy.stats
from scipy.optimize import minimize

def read_dataframe(filename, format):
    df = pd.read_csv('data/' + filename, header=0, index_col=0)/100
    df.index = pd.to_datetime(pd.to_datetime(df.index, format=format))
    df.columns = df.columns.str.strip()
    # print("Dataframe in %: \n", df)
    return df

def annualized_returns(returns, period):
    total_returns = (returns + 1).prod()
    returns_y = total_returns**(period/len(returns)) - 1
    volatility_y = (returns.std() * np.sqrt(period))
    # print("Returns and volatility as a fraction: \n", (returns_y, volatility_y))
    return returns_y, volatility_y

def max_drawdown(returns):
    wealth_index = 100*(1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdown = (wealth_index - previous_peaks)/previous_peaks
    drawdown_idxmin = drawdown.idxmin()
    drawdown_min = drawdown.min()
    print("Max drawdown as a fraction: \n", drawdown.plot.line())
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

def portfolio_returns(weights, returns):
    # weights = np.reshape(weights, (np.shape(weights)[0], 1))
    return weights.T @ returns

def portfolio_volatility(weights, covariance):
    # weights = np.reshape(weights, (np.shape(weights)[0], 1))
    return (weights.T @ covariance @ weights)**0.5

def efficient_frontier_2_asset(returns, n_points):
    returns_y, _ = annualized_returns(returns, 12)
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    returns_p = [portfolio_returns(w, returns_y) for w in weights]
    volatility_p = [portfolio_volatility(w, returns.cov()) for w in weights]
    df_frontier = pd.DataFrame({"Returns": returns_p, "Volatility": volatility_p})
    return df_frontier.plot.line(x="Volatility", y = "Returns", style='.-', figsize=(15, 7))

def minimize_volatility(returns_y, target_return, covariance):
    n = returns_y.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0), )*n

    return_is_target = {
        'type': 'eq',
        'args': (returns_y, ),
        'fun': lambda weights, returns_y: target_return - portfolio_returns(weights, returns_y)
    }

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    results = minimize(
        portfolio_volatility, init_guess,
        args = (covariance, ), method='SLSQP',
        options={'disp': False},
        constraints=(return_is_target, weights_sum_to_1),
        bounds=bounds
    )
    return results.x

def optimal_weights(returns_y, covariance, n_points):
    target_returns = np.linspace(returns_y.min(), returns_y.max(), n_points)
    weights = [minimize_volatility(returns_y, target_return, covariance) for target_return in target_returns]
    return weights

def efficient_frontier_n_asset(returns_y, covariance, n_points):
    weights = optimal_weights(returns_y, covariance, n_points)
    returns = [portfolio_returns(weight, returns_y) for weight in weights]
    volatilitys = [portfolio_volatility(weight, covariance) for weight in weights]
    efficient_frontier = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatilitys
    })
    return efficient_frontier.plot.line(x="Volatility", y="Returns", style='.-', figsize=(15, 7))