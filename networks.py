# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 22:31:21 2015

@author: Ndil Sou
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import networkx as nx
import pickle

import copulas 

from os import listdir

init_path = "/Dataset/SerializedData/"

datafiles_name = listdir(init_path)
begin_end =  '_2010-01-01_2014-12-31'
date = begin_end
dataset = dict()
for name in datafiles_name:
    dump_path = init_path + name
    with open(dump_path, 'rb') as serialiser:
        my_unpickler = pickle.Unpickler(serialiser)
        label = name.split('.')[0]
        dataset[label] = my_unpickler.load()

#%%
###############################################################################
#                       CONSTRUCTION OF THE GRAPH
###############################################################################

market_graph = nx.Graph()

market_graph.add_nodes_from(dataset['fitted_marginal' + date])


#Adding the nodes
for node in market_graph.nodes():
    market_graph.node[node]['fitted_marginal'] = dataset['fitted_marginal' + date][node]
    market_graph.node[node]['garch_data'] = dataset['garch_data' + date][node]
    market_graph.node[node]['returns_data'] = dataset['returns_data' + date ][node]
    market_graph.node[node]['uniform_data'] = dataset['uniform_data'  + date][node]
    market_graph.node[node]['marginal_parameters'] = dataset['marginal_parameters'  + date][node]
    market_graph.node[node]['mean'] = dataset['mean'  + date][node]




#Adding the edges
keys = dataset['copula_parameters_2010-01-01_2014-12-31'].columns.tolist()
values = dataset['copula_parameters_2010-01-01_2014-12-31'].values.flatten()

for idx, pair in enumerate(keys):
    pair = pair.split(':')
    clayton = copulas.Clayton(values[idx])
    weight = 1-clayton.lTDC
    market_graph.add_edge(pair[0], pair[1], 
                          weight=weight,
                          copula=clayton,
                          dependence=clayton.lTDC)


#Now we build a Minimal Spanning Tree out of the complete market graph. 
minimal_market_graph = nx.minimum_spanning_tree(market_graph)

#%%
###############################################################################
#                       GRAPH BUILDER FUNCTIONS
###############################################################################

#We build a random graph on the initial fully connected graph
def generate_random_market_graph(market_graph):
    random_market_graph = market_graph.copy()
    for edge in market_graph.edges(data=True):
        if edge[2]['dependence'] <= stats.uniform.rvs():
            random_market_graph.remove_edge(edge[0], edge[1])
    return random_market_graph

def generate_p_market_graph(market_graph, p):
    random_market_graph = market_graph.copy()
    for edge in market_graph.edges(data=True):
        if edge[2]['dependence'] <= p:
            random_market_graph.remove_edge(edge[0], edge[1])
    return random_market_graph

#Now we build a N-connected graph where each node cannot have more than N edges.
#Edges are chosen based on their weights.
N = 9
def generate_n_market_graph(market_graph, N):
    n_market_graph = market_graph.copy()
    for node in market_graph.nodes():
        edges = n_market_graph[node]
        if len(edges) > N:
            weights = list()
            for key in edges.keys():
                weights.append(edges[key]['dependence'])
            rank = sorted(zip(weights, edges.keys()), reverse=True)
            to_remove = zip(*rank[N:])[1] #We "unzip" the list and retrieve the M - N elt to delete.
            edge_list = [(node, edge) for edge in to_remove]
            n_market_graph.remove_edges_from(edge_list)
    return n_market_graph


def simulate_return(G, node1, node2, date):
    u2, v = stats.uniform.rvs(size=2)
    u1 = G[node1][node2]['copula'].invccdf(u2 ,v )
    r1 = G.node[node1]['fitted_marginal'].invcdf(u1)
    r2 = G.node[node2]['fitted_marginal'].invcdf(u2)
    mean1 = G.node[node1]['mean'].fittedvalues.ix[date]
    vol1 = np.sqrt(G.node[node1]['garch_data'].ix[date])
    r1 = mean1 + vol1 * r1
    mean2 = G.node[node2]['mean'].fittedvalues.ix[date]
    vol2 = np.sqrt(G.node[node2]['garch_data'].ix[date])
    r2 = mean2 + vol2 * r2
    return r1, r2

#%%
###############################################################################
#                       SIMULATION ALGORITHM
###############################################################################

def propagate( hit_node, parent, date, G, returns_dict):
    """
    Propagate a shock from a given node to all its neighbours. 
    Recursive function.
    If the neighbourhood is empty, do nothing.
    
    Keyword arguments:
    hit_node -- Node that propagate the shock.
    parent -- parent of the initial node. Prevent backward loop.
    date -- Date as string. is used to locate the appropriate value for the mean 
        the volatility.
    G -- market graph
    returns_dict -- dictionnary of simulated returns, key is node name.
    
    Behaviour:
    If the neighborhood is not empty the returns_dict is updated with the new 
    simulated returns.
    
    Return:
    (node, status)
    returns the name of the node and the status of the propagation:
    
    1 -- the neighborhood is not empty, returns_dict was updated
    0 -- the neighborhood is empty, returns_dict was not altered.
    """
    filtered_neighborhood = G.neighbors(hit_node)
    r_hit = returns_dict[hit_node]
    if parent:
        filtered_neighborhood.remove(parent)
    if filtered_neighborhood:
        u_hit = G.node[hit_node]['fitted_marginal'].cdf(r_hit)
        for node in filtered_neighborhood:
            v = stats.uniform.rvs()
            u_node = G[hit_node][node]['copula'].invccdf(u_hit , v)
            r_node = G.node[node]['fitted_marginal'].invcdf(u_node)
            mean_node = G.node[node]['mean']
            vol_node = np.sqrt(G.node[node]['garch_data'].ix[date])
            returns_dict[node] = mean_node + vol_node * r_node
            
            propagate(node, hit_node, date, G, returns_dict)
            
        return (hit_node, 1)
    else:
        return (hit_node, 0)
        
        
def simulate_market_graph(hit_node, shock, date, G, N=1):
    """
    Initialize simulation of returns conditionally on a given asset 
    receiving a specified shock.

    Keyword arguments:
    hit_node - key of the initial node.
    shock -- return of the initial node in the scenario.  
    date -- Date as string. is used to locate the appropriate value for the mean 
        the volatility.
    G -- market graph
    N -- number of simulations, default 1
    
    Return:
    node_keys -- keys of all the node in the same order as the columns of the 
    simulation table.
    simulation_table -- (N, G.nb_node)-array contains the simulations. Each row is
    a different simulation. Each column can be mapped to node_keys.
    """
    width = len(G.nodes()) #number of nodes in the graph
    simulation_table = np.empty((N,width))
    for i in np.arange(N):
        returns_dict = {hit_node : shock}
        propagate( hit_node, None, date, G, returns_dict)
        simulation_table[i,:] = returns_dict.values()
    return returns_dict.keys(), simulation_table
 
 #%%
###############################################################################
#                       EXPERIMENT
###############################################################################

assets, simulations = simulate_market_graph('S5AEROX',-0.3,'2014-12-31',minimal_market_graph, N=1000)
nb_asset = len(assets)
equal_ptf = np.ones((nb_asset,1))
ptf_returns = np.dot(simulations,equal_ptf)
plt.figure(1)
plt.hist(ptf_returns, bins = 50)

#%%
###############################################################################
#                       UTILITIES
###############################################################################

G = minimal_market_graph
hit_node = u'S5CPGS'
parent = u'S5AEROX'
r_hit = -0.1
returns_dict = dict()
date = '2014-12-31'
   
plt.figure(2)
nx.draw_spring(minimal_market_graph, with_labels=True,
                node_shape='h', node_size=1900, font_size =9, scale=250, alpha=0.9)

alpha = 2
sim_clayton = copulas.Clayton(alpha)
sim_gumbel = copulas.Gumbel(alpha)
U = stats.uniform.rvs(size=(800,2))
valC = sim_clayton.invccdf(U[:,0],U[:,1])
valG = np.array(U[:,0])
for i in range(len(valG)):
    valG[i] = sim_gumbel.invccdf(U[i,0],U[i,1])
plt.figure(3)    
plt.ylim([0,1])
plt.xlim([0,1])
plt.scatter(valC,U[:,0])
plt.scatter(U[:,0],valG)






#%%
#PLOT HISTOGRAM OF STANDARDISED RETURNS

node1 = 'S5HOTRX'
label1 = 'Hotels, Restaurants & Leisure'
node2 = 'S5INSSX'
label2 = 'Internet Software & Services'

marginal_S5HOTRX = market_graph.node[node1]['fitted_marginal']
marginal_S5INSSX = market_graph.node[node2]['fitted_marginal']
U = market_graph.node[node1]['uniform_data']
R_S5INSSX = [float(marginal_S5INSSX.invcdf(u)) for u in U]
U = market_graph.node[node2]['uniform_data']
R_S5HOTRX = [float(marginal_S5HOTRX.invcdf(u)) for u in U]

fig = plt.figure(figsize=(10,5))

ax1 = plt.subplot(1, 2, 1)
ax1.set_title('Histogram of returns for \n {}'.format(label1))
plt.hist(R_S5HOTRX, bins = 50, alpha=0.3, color='r')


plt.xlabel('Standardized Returns')
plt.ylabel('frequency')
plt.ylim([0,600])
plt.xlim([-9,9])  
df = market_graph.node[node1]['fitted_marginal'].df
skew = market_graph.node[node1]['fitted_marginal'].skew

plt.text(-8, 400, '$v={:.3f}$,\n $\lambda={:.3f}$'.format(df, skew), fontsize=15)

ax2 = plt.subplot(1, 2, 2)
ax2.set_title('Histogram of returns for \n {}'.format(label2))

plt.hist(R_S5INSSX, bins = 50)

plt.xlabel('Standardized Returns')
plt.ylabel('frequency')
plt.ylim([0,600])
plt.xlim([-9,9]) 
df = market_graph.node[node2]['fitted_marginal'].df
skew = market_graph.node[node2]['fitted_marginal'].skew

plt.text(-8, 400, '$v={:.3f}$,\n $\lambda={:.3f}$'.format(df, skew), fontsize=15)


#TABLE TAIL EXPONENTS
col = ['Tail Exponent', 'CI lower bound', 'CI Upper Bound', 'Scaling Factor']
tail10_table = pd.DataFrame(columns=col, index=market_graph.nodes())
for node in market_graph.nodes():
    tail_list = market_graph.node[node]['marginal_parameters']['tail10']
    tail10_table['Tail Exponent'].ix[node] = float(tail_list[0])
    tail10_table['CI lower bound'].ix[node] = float(tail_list[2][0])
    tail10_table['CI Upper Bound'].ix[node] = float(tail_list[2][1])
    tail10_table['Scaling Factor'].ix[node] = float(tail_list[1])

#TABLE MARGINALS
col = ['alpha', 'beta','omega','theta', 'v', 'skew']
marginal_table = pd.DataFrame(columns=col, index=market_graph.nodes())
for node in market_graph.nodes():
     marginal_table['alpha'].ix[node] = market_graph.node[node]['marginal_parameters']['alpha']  
     marginal_table['beta'].ix[node] = market_graph.node[node]['marginal_parameters']['beta'] 
     marginal_table['omega'].ix[node] = market_graph.node[node]['marginal_parameters']['omega'] 
     marginal_table['theta'].ix[node] = market_graph.node[node]['marginal_parameters']['theta'] 
     marginal_table['v'].ix[node] = market_graph.node[node]['marginal_parameters']['df'] 
     marginal_table['skew'].ix[node] = market_graph.node[node]['marginal_parameters']['skew'] 
     
     
#TABLE COPULA
col = ['alpha', 'lTDC']
index = [':'.join(edge) for edge in minimal_market_graph.edges()]
copulas_table = pd.DataFrame(columns=col, index=index)
for edge in minimal_market_graph.edges():
    node1, node2 = edge    
    copulas_table['alpha'].ix[':'.join(edge)] = minimal_market_graph[node1][node2]['copula'].alpha
    copulas_table['lTDC'].ix[':'.join(edge)] = minimal_market_graph[node1][node2]['copula'].lTDC
    
#PLOT ARMA VS SERIES
plotter = pd.concat((market_graph.node[node]['returns_data']
                    , market_graph.node[node]['mean']), axis = 1)
plotter.columns = ['Initial Returns Series', 'Arma Prediction']
plotter.plot(title='Comparison of ARMA prediction with original series for {}'.format(node))

#PLOT GARCH VS SERIES
plotter = pd.concat((market_graph.node[node]['returns_data']
                    , np.sqrt(market_graph.node[node]['garch_data'])), axis = 1)
plotter.columns = ['Initial Returns Series', 'A-GARCH Filter']
ax = plotter.plot(title='Comparison of A-GARCH prediction with original series for {}'.format(node))
ax.fill_between(plotter.index,np.sqrt(market_graph.node[node]['garch_data']), alpha=0.9,color='r')
