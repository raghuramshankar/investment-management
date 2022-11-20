# %%
import os
import sys

if '__ipython__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'funcs'))
    %load_ext autoreload
    %autoreload 2

import risk_kit as rk
import pandas as pd
import numpy as np

df = rk.read_dataframe('ind30_m_vw_rets.csv', "%Y%m")

"""plot efficient frontiers for 2 weight portfolios"""
returns_m = df["1996":"2000"].get(["Games", "Beer"])

_ = rk.efficient_frontier(returns_m)