# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:28:22 2015

@author: Ndil Sou
"""

import pandas as pd
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import statsmodels.api as sm

import pickle

import copulas


init_path = "/Dataset/"
fullset = "DATASETLVL3MULTIWS.xlsm"
testset = "trainset.xlsm"
#Choose to estimate all (fullset) or just 3 series (trainset)
path = init_path + fullset

#gics_path = init_path + 'gics_dict.csv'
#gics_map = pd.read_csv(gics_path, sep=';', index_col=0,header=None)
file_out = lambda x: init_path + 'res/' + x + '.tex'

#%%
###############################################################################
#                       ESTIMATION OF THE TAIL EXPONENT
###############################################################################
#Now we will need to :
#1) remove any empty series from the dictionaries. 
#2) rank the entries for each serie and perform the log-log rank regression to 
#retrieve the tail index.

def tail_exponent(series, cutoff, side):
    """
    input series: a ndarray of pandas Series
    cutoff : a value in (0,1) represent the percentage of the tail 
    that we work on.
    side : 0 (Left tail) or 1 (right Tail)
    
    we estimate the regression:
    log(rank-1/2) = a + b * log(Size)
    the tail_exponent is -b
    returns:
    tail_exponent_value; log_scaling_const; CI
    """
    length = len(series)
    if side == 0:
        index = np.where(series <= 0)
    else:
        index = np.where(series > 0)
    if isinstance(series, (pd.Series, pd.DataFrame)):
        series = series.iloc[index[0]].values
    if isinstance(series, list):
        series = np.array(series)
    series.shape = (length,1)
        
    series = np.abs(series)
    tail_cutoff = int(cutoff*length)
    series = sorted(series, reverse=True)[:tail_cutoff]
    length = len(series)
    rank = np.array(range(1, length+1))
    rank.shape = (len(rank),1)

    const = np.ones((length,1))    
    X = np.concatenate((np.log(series), const), axis = 1)
    Y = np.log(rank - 0.5)
    tail_exp, C = lstsq(X,Y)[0]
    tail_exp = -tail_exp

    CI = 1.96 * np.sqrt(2.0/length)
    conf_interval = [tail_exp*(1 - CI), tail_exp*(1 + CI)]
    return tail_exp, C, conf_interval
    

    

#%%
###############################################################################
#                       CALIBRATION OF MARGINAL SKEW-T DISTRIBUTION
###############################################################################
print('EXTRACTION OF TIME SERIES')


xls_file = pd.ExcelFile(path)
sheets = xls_file.sheet_names
sheets = [sheet.replace(' Index', '') for sheet in sheets]
#Now we need to get rid of the sheets named Data and GICS, they are not 
#directly relevant here.
sheets = [sheet for sheet in sheets if (sheet != 'GICS' and sheet != 'Data')]



len_cutoff = 3778
#First sweep, Extract Transform Load
price_data = pd.DataFrame()
clean_sheets = list()
for sheet in sheets:
    print('\n' + sheet)
    raw_price_data = xls_file.parse(''.join((sheet,' Index')), 
                             index_col=0)
    #First we drop the volume columns, the study focus is on the prices.
    del raw_price_data['PX_VOLUME']
    init_len = len_cutoff
    final_len = float(len(raw_price_data['PX_LAST'].dropna()))
    pct_valid = final_len / init_len
    if (not raw_price_data.dropna().empty) and (pct_valid >= 0.95):
        print('OK')
        raw_price_data.columns = [sheet]
        price_data = pd.concat((price_data, raw_price_data), axis = 1)
        clean_sheets.append(sheet)
    else:
        print('KO: Not added to final set')

sheets = clean_sheets

returns_data = np.log(price_data).diff().dropna()
begin = '2010-01-01'
end = '2014-12-31'
returns_data.ix[begin:end]

mean_dict = dict.fromkeys(sheets)

garch_data = pd.DataFrame(columns=sheets,
                          index=returns_data.index)
uniform_data = pd.DataFrame(columns=sheets,
                          index=returns_data.index)
                          
#Fields for each series : alpha, beta, omega, theta, df, skew, tail05, tail10
marginal_parameters = pd.DataFrame(columns=sheets,
                          index=['alpha', 'beta', 'omega', 'theta', 'df', \
                          'skew', 'mean', 'tail05', 'tail10']) 

print('CALIBRATION OF MARGINAL SKEW-T DISTRIBUTION')

fitted_marginal = dict()

#Second Sweep, marginal fitting
p, q = 2, 1
for sheet in sheets:
    print('\n' + sheet)
    #Second we need to make sure that the dataset is ordered correctly in \
    # in chronological order.
    return_frame = returns_data[sheet]
    r_mean = return_frame.mean()
    #arma_mean = sm.tsa.ARMA(return_frame, (p,q)).fit()
    mean_dict[sheet] = r_mean
    return_frame = return_frame - r_mean
    R = return_frame.values.flatten().tolist()
   
    
    #Note that we embedded the constraints in the likelihood functions, therefore 
    #their we can use an unconstrained algorithm.
    x0 = [3, -0.5, 0.1, 0.85, 5e-6, 0.5]
    res = copulas.fit_marginal(R, x0, verbose=0)
    res['tail10'] = tail_exponent(R, 0.1, 0)
    res['tail05'] = tail_exponent(R, 0.05, 0)
    res['mean'] = r_mean
    marginal_parameters[sheet].ix[res.keys()] = res.values()
    
    agarch = copulas.maker_garch(res['alpha'], res['beta'],
                                 res['omega'], res['theta'])
    var_garch = [np.var(R)]
    for i, elt in enumerate(R[:-1]):
        var_garch.append(agarch(elt, var_garch[i]))
    garch_data[sheet] = var_garch
    #Now we extract the uniform data from the adjusted series:
    fitted_marginal[sheet] = copulas.Skew_t(res['df'], res['skew'])
    uniform_data[sheet] = map(fitted_marginal[sheet].cdf, R/np.sqrt(var_garch))

#%%
###############################################################################
#                       CALIBRATION OF COPULA PAIRS
###############################################################################
print('CALIBRATION OF COPULA PAIRS')
#We have N(N-1)/2 Copula Pair to setup.
#Again, we store them in a dictionary. keys naming convention: IDindex1:IDindex2
copula_parameters = pd.DataFrame(index=['alpha'])
upper_sheets = list(sheets)
inner_sheets = list(sheets)
for sheet1 in upper_sheets:
    inner_sheets.remove(sheet1)
    u1 = uniform_data[sheet1]
    for sheet2 in inner_sheets:
        u2 = uniform_data[sheet2]
        joint_key = ":".join((sheet1, sheet2))
        print('\n' + joint_key)
        res = copulas.fit_copula(u1, u2, verbose=False)
        copula_parameters[joint_key] = res


print('\nSerialisation')
#To finish we serialise the dataset for use by the other modules
datadump = {'price_data': price_data,
            'returns_data': returns_data,
            'garch_data': garch_data,
            'uniform_data': uniform_data,
            'copula_parameters': copula_parameters,
            'marginal_parameters': marginal_parameters,
            'fitted_marginal': fitted_marginal,
            'mean': mean_dict
            }


for datatype, dataset in datadump.items():
    dump_path = init_path + 'SerializedData/' + datatype + '_'  + begin + '_' + end + '.dta'
    with open(dump_path, 'wb') as serialiser:
        my_pickler = pickle.Pickler(serialiser)
        my_pickler.dump(dataset)
print('DONE')

#with open('prices_dataframe', 'wb') as serialiser:
#    my_pickler = pickle.Pickler(serialiser)
#    my_pickler.dump(price_data)
