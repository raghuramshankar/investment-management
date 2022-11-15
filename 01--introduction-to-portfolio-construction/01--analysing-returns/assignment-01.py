# %%
import os
import sys

if '__ipython__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), '..', 'funcs'))

import scipy.stats
import funcs.risk_kit as rk

df = rk.read_dataframe('Portfolios_Formed_on_ME_monthly_EW.csv', "%Y%m")
returns_m = df.get(["Lo 20", "Hi 20"])/100
returns_slice_m = df["1999":"2015"].get(["Lo 20", "Hi 20"])/100

returns_y, volatility_y = rk.annualized_from_monthly(returns_m)
returns_slice_y, volatility_slice_y = rk.annualized_from_monthly(returns_slice_m)
drawdown_slice_idxmin, drawdown_slice_min = rk.max_drawdown(returns_slice_m)

df = rk.read_dataframe('edhec-hedgefundindices.csv', '%d/%m/%Y')
returns_m = df["2009":"2018"]
negative_semi_deviation = returns_m[returns_m<0].std(ddof=0)
skewness = rk.skew(returns_m)
kurtosis = rk.kurtosis(returns_m)

"""plot and print"""
print("1: Lo 20 Yearly returns =", round(returns_y["Lo 20"], 2), "%")
print("2: Lo 20 Yearly volatility =", round(volatility_y["Lo 20"], 2), "%")
print("3: Hi 20 Yearly returns =", round(returns_y["Hi 20"], 2), "%")
print("4: Hi 20 Yearly volatility =", round(volatility_y["Hi 20"] ,2), "%")
print("5: Lo 20 Yearly returns 1999-2015 =", round(returns_slice_y["Lo 20"], 2), "%")
print("6: Lo 20 Yearly volatility 1999-2015 =", round(volatility_slice_y["Lo 20"], 2), "%")
print("7: Hi 20 Yearly returns 1999-2015 =", round(returns_slice_y["Hi 20"], 2), "%")
print("8: Hi 20 Yearly volatility 1999-2015 =", round(volatility_slice_y["Hi 20"], 2), "%")
print("9: Lo 20 Max drawdown value 1999-2015 =", - round(drawdown_slice_min["Lo 20"], 2), "%")
print("10: Lo 20 Max drawdown date 1999-2015 =", (drawdown_slice_idxmin["Lo 20"]).strftime("%Y-%m"))
print("11: Hi 20 Max drawdown value 1999-2015 =", - round(drawdown_slice_min["Hi 20"], 2), "%")
print("12: Hi 20 Max drawdown date 1999-2015 =", (drawdown_slice_idxmin["Hi 20"]).strftime("%Y-%m"))
print("13: Highest semi deviation among hedge funds =", negative_semi_deviation.idxmax())
print("14: Lowest semi deviation among hedge funds =", negative_semi_deviation.idxmin())
print("15: Most negative skewness among hedge funds =", skewness.idxmin())
print("16: Highest kurtosis among hedge funds =", skewness.idxmax())