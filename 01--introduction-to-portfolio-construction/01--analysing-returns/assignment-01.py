# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('01--introduction-to-portfolio-construction/data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0)
# df = pd.read_csv('../data/Portfolios_Formed_on_ME_monthly_EW.csv', header=0, index_col=0)
df.index = pd.to_datetime(pd.to_datetime(df.index, format="%Y%m"))
returns_m = df.get(["Lo 20", "Hi 20"])/100

"""questions 1-4"""
total_returns_m = (returns_m + 1).prod() - 1
returns_y = ((total_returns_m + 1)**(12/len(returns_m)) - 1) * 100
volatility_y = (returns_m.std() * np.sqrt(12))* 100

"""questions 5-12"""
returns_slice_m = df["1999":"2015"]/100
total_returns_slice_m = (returns_slice_m + 1).prod() - 1
returns_slice_y = ((total_returns_slice_m + 1)**(12/len(returns_slice_m)) - 1) * 100
volatility_slice_y = (returns_slice_m.std() * np.sqrt(12))* 100

"""plot and print"""
print("1: Lo 20 Yearly returns =", returns_y["Lo 20"], "%")
print("2: Lo 20 Yearly volatility =", volatility_y["Lo 20"], "%")
print("3: Hi 20 Yearly returns =", returns_y["Hi 20"], "%")
print("4: Hi 20 Yearly volatility =", volatility_y["Hi 20"], "%")
print("5: Lo 20 Yearly returns 1999-2015 =", returns_slice_y["Lo 20"], "%")
print("6: Lo 20 Yearly volatility 1999-2015 =", volatility_slice_y["Lo 20"], "%")
print("7: Hi 20 Yearly returns 1999-2015 =", returns_slice_y["Hi 20"], "%")
print("8: Hi 20 Yearly volatility 1999-2015 =", volatility_slice_y["Hi 20"], "%")

returns_m.plot()
# plt.show()
# %%