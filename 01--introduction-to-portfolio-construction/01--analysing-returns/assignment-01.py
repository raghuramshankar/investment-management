# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# df = pd.read_csv('01--introduction-to-portfolio-construction/data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0)
df = pd.read_csv('../data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0)
df.index = pd.to_datetime(pd.to_datetime(df.index, format="%Y%m"))
returns_m = df.get(["Lo 20", "Hi 20"])/100
returns_slice_m = df["1999":"2015"]/100

"""questions 1-4"""
total_returns_m = (returns_m + 1).prod() - 1
returns_y = ((total_returns_m + 1)**(12/len(returns_m)) - 1) * 100
volatility_y = (returns_m.std() * np.sqrt(12))* 100

"""questions 5-12"""
total_returns_slice_m = (returns_slice_m + 1).prod() - 1
returns_slice_y = ((total_returns_slice_m + 1)**(12/len(returns_slice_m)) - 1) * 100
volatility_slice_y = (returns_slice_m.std() * np.sqrt(12))* 100
wealth_index_slice = 100*(1 + returns_slice_m.get(["Lo 20", "Hi 20"])).cumprod()
previous_peaks_slice = wealth_index_slice.cummax()
drawdown_slice = (wealth_index_slice - previous_peaks_slice)/previous_peaks_slice
drawdown_slice_idxmin = drawdown_slice.idxmin()
drawdown_slice_min = drawdown_slice.min() * 100

"""plot and print"""
print("1: Lo 20 Yearly returns =", round(returns_y["Lo 20"], 2), "%")
print("2: Lo 20 Yearly volatility =", round(volatility_y["Lo 20"], 2), "%")
print("3: Hi 20 Yearly returns =", round(returns_y["Hi 20"], 2), "%")
print("4: Hi 20 Yearly volatility =", round(volatility_y["Hi 20"] ,2), "%")
print("5: Lo 20 Yearly returns 1999-2015 =", round(returns_slice_y["Lo 20"], 2), "%")
print("6: Lo 20 Yearly volatility 1999-2015 =", round(volatility_slice_y["Lo 20"], 2), "%")
print("7: Hi 20 Yearly returns 1999-2015 =", round(returns_slice_y["Hi 20"], 2), "%")
print("8: Hi 20 Yearly volatility 1999-2015 =", round(volatility_slice_y["Hi 20"], 2), "%")
print("9: Lo 20 Max drawdown value 1999-2015 =", round(drawdown_slice_min["Lo 20"], 2), "%")
print("10: Lo 20 Max drawdown date 1999-2015 =", (drawdown_slice_idxmin["Lo 20"]).strftime("%Y-%m"))
print("11: Hi 20 Max drawdown value 1999-2015 =", round(drawdown_slice_min["Hi 20"], 2), "%")
print("12: Hi 20 Max drawdown date 1999-2015 =", (drawdown_slice_idxmin["Hi 20"]).strftime("%Y-%m"))

returns_m.plot()

# plt.show()
# %%