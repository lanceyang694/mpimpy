## PageRank 

'''

@author: Ling Yang
A simulation demonstration of PIM for PageRank

'''
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from mpimpy import memmatdp

## original state transition probability matrix
np.random.seed(42)
size = 16
t_mat = np.abs(np.random.rand(size, size))
t_mat = t_mat * np.random.randint(0, 2, size=(size, size))
t_mat = t_mat * (np.ones((size, size)) - np.diag(np.ones(size)))
t_mat = np.dot(t_mat, np.diag(1 / np.sum(t_mat, axis=0)))

## initial probability vector
x_0 = np.ones(t_mat.shape[0]) / t_mat.shape[0]

d = 0.85  # damping coefficient
a_mat = np.dot(d, t_mat) + (1 - d) / t_mat.shape[0]  # transition matrix

## define the hardware parameters
dpe_dp = memmatdp.diffpairdpe(HGS=1/1.3e5, LGS=1/2.23e6, g_level=16, var=0.02, vnoise = 0, wire_resistance=2.93,
                            rdac=256, radc=1024, vread=0.1, array_size=(32, 32))

def page_rank(A, x_0, min_delta= 1e-6, max_iter=100):
    
    count = 0
    
    while(count < max_iter):  
        
        # y = np.dot(A, x_0)                    # Ideal solution
        y = dpe_dp.MapReduceDot(x_0, A.T).T    # PIM solution
        x_1 = y / np.max(y)
        e = np.max(np.abs(x_1-x_0))
        
        if(e < min_delta):
            break
        
        x_0 = x_1
        count += 1
        
    print(x_0)
    print(np.argsort(x_0))
    print('Iteration number: ', count)
    
    return x_1

pr = page_rank(a_mat, x_0)


def show_graph(A, pr):
    
    df = pd.DataFrame(A * (np.ones((size, size)) - np.diag(np.ones(size))))
    links = df.stack().reset_index()
    links.columns = ['node1', 'node2','weight']
    links_filtered = links.loc[ (links['weight'] > 0.0) & (links['node1'] != links['node2']) ]
    graph = nx.from_pandas_edgelist(links_filtered, 'node1', 'node2', create_using=nx.DiGraph())
    
    positions=nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, positions, node_color='#75bbfd', node_size=pr*500, alpha=0.8)
    nx.draw_networkx_edges(graph, positions, edge_color='k', width=1., alpha=0.2)
    nx.draw_networkx_labels(graph, positions, font_size=10)

    plt.show()

show_graph(a_mat, pr=pr)

