# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

import numpy as np
import scipy.stats as stats


init_path = 'C:/SourceRepo/Research/Dataset'
file_name = 'trainset.xlsm'

path = '/'.join([init_path, file_name])
dataset = pd.read_excel(path, 
              index_col=0,
              sheetname=0)
dataset.sort_index(axis=0, inplace=True)

del dataset['PX_VOLUME']
data_returns = np.log(dataset).diff().dropna()

arma_mod11 = sm.tsa.ARMA(data_returns, (1,1)).fit()
qqplot(arma_mod11.resid, line='q', fit=True)
