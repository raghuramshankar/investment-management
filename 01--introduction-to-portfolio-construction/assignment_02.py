# %%
import os
import sys

if '__ipython__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(os.getcwd()))))
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'funcs'))
    %load_ext autoreload
    %autoreload 2

import risk_kit as rk

df = rk.read_dataframe('ind30_m_vw_rets.csv', "%Y%m")