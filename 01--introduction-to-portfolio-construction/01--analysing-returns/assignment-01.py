# %%
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('01--introduction-to-portfolio-construction/data/Portfolios_Formed_on_ME_monthly_EW.csv')
# df = pd.read_csv('../data/Portfolios_Formed_on_ME_monthly_EW.csv')

"""question 1, 2, 3 and 4"""
data = df.get(['Lo 20', 'Hi 20'])
data = data/100
total_returns_m = (data + 1).prod() - 1
returns_y = (total_returns_m + 1)**(12/len(data)) - 1
volatility_y = data.std() * np.sqrt(12)


"""plot and print"""
# print("Total monthly returns (%) = \n", total_returns_m * 100)
print("Yearly returns (%) = \n", returns_y * 100)
print("Yearly volatility = \n", volatility_y)
data.plot()
# plt.show()