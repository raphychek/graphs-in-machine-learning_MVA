"""
Functions to build and visualize similarity graphs, and to choose epsilon in epsilon-graphs.
"""

import numpy as np
import matplotlib.pyplot as pyplot
import scipy.spatial.distance as sd
import sys
import os

from utils import plot_graph_matrix, min_span_tree
from generate_data import worst_case_blob, blobs, two_moons, point_and_circle


def build_similarity_graph(X, var=1.0, eps=0.0, k=0):
    
    """
    TO BE COMPLETED.

    Computes the similarity matrix for a given dataset of samples. If k=0, builds epsilon graph. Otherwise, builds
    kNN graph.

    :param X:    (n x m) matrix of m-dimensional samples
    :param var:  the sigma value for the exponential function, already squared
    :param eps:  threshold eps for epsilon graphs
    :param k:    the number of neighbours k for k-nn. If zero, use epsilon-graph
    :return:
        W: (n x n) dimensional matrix representing the adjacency matrix of the graph
    """
    n = X.shape[0]
    W = np.zeros((n, n))

    """
    Build similarity graph, before threshold or kNN
    similarities: (n x n) matrix with similarities between all possible couples of points.
    The similarity function is d(x,y)=exp(-||x-y||^2/(2*var))
    """
  
    #similarities = np.zeros((n, n))
    
    similarities = sd.cdist(X,X,'euclidean')
    similarities = np.exp(-similarities*similarities/(2*var))
    

    # If epsilon graph
    if k == 0:
        """
        compute an epsilon graph from the similarities             
        for each node x_i, an epsilon graph has weights             
        w_ij = d(x_i,x_j) when w_ij >= eps, and 0 otherwise          
        """
        for i in range(n):
            for j in range(n):
                if i!=j and similarities[i][j]>=eps:
                    W[i][j]=similarities[i][j]
                    W[j][i]=similarities[j][i]

    # If kNN graph
    if k != 0:
        """
        compute a k-nn graph from the similarities                   
        for each node x_i, a k-nn graph has weights                  
        w_ij = d(x_i,x_j) for the k closest nodes to x_i, and 0     
        for all the k-n remaining nodes                              
        Remember to remove self similarity and                       
        make the graph undirected                                    
        """
        tmp = []
        for i in range(n):
            tmp = [ [j,np.copy(similarities[i][j])] for j in range(n)]
            tmp = sorted(tmp, key = lambda x : x[1], reverse = True)
            for l in range(1,k+1):
                W[i][tmp[l][0]]=tmp[l][1]
                W[tmp[l][0]][i]=tmp[l][1]
    return W


def plot_similarity_graph(X, Y, var=1.0, eps=0.0, k=5):
    """
    Function to plot the similarity graph, given data and parameters.

    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n, ) vector with cluster assignments
    :param var:  the sigma value for the exponential function, already squared
    :param eps:  threshold eps for epsilon graphs
    :param k:    the number of neighbours k for k-nn
    :return:
    """
    # use the build_similarity_graph function to build the graph W
    # W: (n x n) dimensional matrix representing the adjacency matrix of the graph
    W = build_similarity_graph(X, var, eps, k)

    # Use auxiliary function to plot
    plot_graph_matrix(X, Y, W)


def how_to_choose_epsilon(gen_pam,k):
    """
    TO BE COMPLETED.

    Consider the distance matrix with entries dist(x_i, x_j) (the euclidean distance between x_i and x_j)
    representing a fully connected graph.
    One way to choose the parameter epsilon to build a graph is to choose the maximum value of dist(x_i, x_j) where
    (i,j) is an edge that is present in the minimal spanning tree of the fully connected graph. Then, the threshold
    epsilon can be chosen as exp(-dist(x_i, x_j)**2.0/(2*sigma^2)).
    """
    # the number of samples to generate
    num_samples = 100

    # the option necessary for worst_case_blob, try different values
    #gen_pam = 10  # to understand the meaning of the parameter, read worst_case_blob in generate_data.py

    # get blob data
    # X, Y = worst_case_blob(num_samples, gen_pam)
    X, Y = two_moons(num_samples)
    

    """
     use the distance function and the min_span_tree function to build the minimal spanning tree min_tree                   
     - var: the exponential_euclidean's sigma2 parameter          
     - dists: (n x n) matrix with euclidean distance between all possible couples of points                   
     - min_tree: (n x n) indicator matrix for the edges in the minimal spanning tree                           
    """
    var = 1.0
    dists = sd.cdist(X,X,'euclidean')  # dists[i, j] = euclidean distance between x_i and x_j
    
    min_tree = min_span_tree(dists)
    
    l = []
    n1,m1 = min_tree.shape
    for i in range(n1):
        for j in range(m1):
            if min_tree[i][j]==True:
                l.append([(i,j),dists[i][j]])
    l = sorted(l, key = lambda x : x[1], reverse = True)

    #print(min_tree)

    """
    set threshold epsilon to the max weight in min_tree 
    """
    distance_threshold = l[0][1]
    eps = np.exp(-distance_threshold**2.0/(2*var))

    """
    use the build_similarity_graph function to build the graph W  
     W: (n x n) dimensional matrix representing                    
        the adjacency matrix of the graph
       use plot_graph_matrix to plot the graph                    
    """
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    plot_graph_matrix(X, Y, W)
    return eps,X,Y,W


#if __name__ == '__main__':
#    for gp in [0,1,10,100]:
#        print(gp)
#        how_to_choose_epsilon(gp,0)
#    for k in [0,1,2,5,10]:
#        how_to_choose_epsilon(0,k)