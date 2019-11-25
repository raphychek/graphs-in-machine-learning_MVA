import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.spatial.distance as sd
from scipy.io import loadmat
import os

from helper import build_similarity_graph, build_laplacian, plot_classification, label_noise, \
                    plot_classification_comparison, plot_clusters, plot_graph_matrix

from scipy.optimize import minimize
from autograd import jacobian

np.random.seed(50)


def build_laplacian_regularized(X, laplacian_regularization=0, var=1.0, eps=0.0, k=0, laplacian_normalization=""):
    """
    Function to construct a regularized Laplacian from data.

    :param X: (n x m) matrix of m-dimensional samples
    :param laplacian_regularization: regularization to add to the Laplacian (parameter gamma)
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: Q (n x n ) matrix, the regularized Laplacian
    """
    # build the similarity graph W
    W = build_similarity_graph(X, var, eps, k)

    """
    Build the Laplacian L and the regularized Laplacian Q.
    Both are (n x n) matrices.
    """
    L = build_laplacian(W, laplacian_normalization)

    # compute Q
    Q = L + laplacian_regularization*np.eye(W.shape[0])

    return Q


def mask_labels(Y, l):
    """
    Function to select a subset of labels and mask the rest.

    :param Y:  (n x 1) label vector, where entries Y_i take a value in [1, ..., C] , where C is the number of classes
    :param l:  number of unmasked (revealed) labels to include in the output
    :return:  Y_masked:
               (n x 1) masked label vector, where entries Y_i take a value in [1, ..., C]
               if the node is labeled, or 0 if the node is unlabeled (masked)
    """
    num_samples = np.size(Y, 0)

    """
     randomly sample l nodes to remain labeled, mask the others   
    """
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    indices = indices[:l]

    Y_masked = np.zeros(num_samples)
    Y_masked[indices] = Y[indices]

    return Y_masked

def hard_hfs_online_ssl(L,W,Y):
    num_samples = np.size(Y)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    l_idx, u_idx = [], []
    for i in range(num_samples):
        if Y[i]==0:
            u_idx.append(i)
        else:
            l_idx.append(i)

    num_labels = len(l_idx)
            
    Luu = L[num_labels:num_samples,num_labels:num_samples]
    inv_Luu = np.linalg.inv(Luu)
    
    Wul = W[num_labels:num_samples,0:num_labels]
    
    f_l = np.zeros((len(l_idx),num_classes))
    for i in l_idx:
        f_l[i][int(Y[i])-1]=1
         
    # this is the closed form solution, showed in class
    f_u = inv_Luu @ (Wul @ f_l)

    #print("f_l",f_l)
    #print("f_u",f_u)

    labels = np.zeros((num_samples,num_classes))
    for i in range(len(f_l)):
        lab = np.argmax([f_l[i][k] for k in range(num_classes)])+1
        labels[i][lab-1]=1
    for i in range(len(f_u)):
        for j in range(num_classes):
            labels[i+len(f_l)][j]=f_u[i][j]
    #labels = np.concatenate(labels,f_u)
    #print("labels",labels)
            
    return labels
    

def hard_hfs(X, Y, laplacian_regularization, var=1, eps=0, k=6, laplacian_normalization=""):
    """
    Function to perform hard (constrained) HFS.
    
    /!\ WE SUPPOSE HERE THAT X AND Y ARE SORTED ACCORDING TO Y /!\

    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [0, 1, ... , num_classes] (0 is unlabeled)
    :param laplacian_regularization: regularization to add to the Laplacian
    =param num_label: the number of kept labels
    :param var: the sigma value for the exponential function, already squared
    :param eps: threshold eps for epsilon graphs
    :param k: number of neighbours k for k-nn. If zero, use epsilon-graph
    :param laplacian_normalization: string selecting which version of the laplacian matrix to construct
                                    'unn':  unnormalized,
                                    'sym': symmetric normalization
                                    'rw':  random-walk normalization
    :return: labels, class assignments for each of the n nodes
    """
    
    num_samples = np.size(X, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1

    """
    Build the vectors:
    l_idx = (l x num_classes) vector with indices of labeled nodes
    u_idx = (u x num_classes) vector with indices of unlabeled nodes
    """    

    l_idx, u_idx = [], []
    for i in range(num_samples):
        if Y[i]==0:
            u_idx.append(i)
        else:
            l_idx.append(i)

    num_labels = len(l_idx)
    print("labels",num_labels)
    print("samples",num_samples)
    

    """
    Compute the hfs solution, remember that you can use the functions build_laplacian_regularized and 
    build_similarity_graph    
    
    f_l = (l x num_classes) hfs solution for labeled data. It is the one-hot encoding of Y for labeled nodes.   
    
    example:         
        if Cl=[0,3,5] and Y=[0,0,0,3,0,0,0,5,5], then f_l is a 3x2  binary matrix where the first column codes 
        the class '3'  and the second the class '5'.    
    
    In case of 2 classes, you can also use +-1 labels      
        
    f_u = array (u x num_classes) hfs solution for unlabeled data
    
    f = array of shape(num_samples, num_classes)
    """
    
    L = build_laplacian_regularized(X,k=k)
        
    Luu = L[num_labels:num_samples,num_labels:num_samples]
    print(Luu.shape)
    print(Luu)
    inv_Luu = np.linalg.inv(Luu)
    
    W = build_similarity_graph(X,k)
    Wul = W[num_labels:num_samples,0:num_labels]
    
    
    f_l = np.zeros((len(l_idx),num_classes))
    for i in l_idx:
        f_l[i][int(Y[i])-1]=1
         
    # this is the closed form solution, showed in class
    f_u = inv_Luu @ (Wul @ f_l)
    
    
    """
    compute the labels assignment from the hfs solution   
    labels: (n x 1) class assignments [1,2,...,num_classes]    
    """
    labels = []
    for i in range(len(f_l)):
        lab = np.argmax([f_l[i][k] for k in range(num_classes)])+1
        labels.append(lab)
    for i in range(len(f_u)):
        lab = np.argmax([f_u[i][k] for k in range(num_classes)])+1
        labels.append(lab)
            
    return labels

def sorting(X,Y_masked,Y):
    yx = sorted(zip(Y_masked,Y,X), key=lambda pair: pair[0], reverse=True)
    new_X = np.array([x for _,_,x in yx])
    new_Y = np.array([y for y,_,_ in yx])
    Y = np.array([y for _,y,_ in yx])
    return new_X,new_Y,Y

def two_moons_hfs():
    """
    HFS for two_moons data.
    """

    """
    Load the data. At home, try to use the larger dataset (question 1.2).    
    """
    # load the data
    in_data = loadmat(os.path.join('data', 'data_2moons_hfs.mat'))
    #in_data = loadmat(os.path.join('data', 'data_2moons_hfs_large.mat'))
    X = in_data['X']
    Y = in_data['Y'].squeeze()
    #print(Y)

    # automatically infer number of labels from samples
    num_samples = np.size(Y, 0)
    num_classes = len(np.unique(Y))

    """
    Choose the experiment parameters
    """
    var = 1
    eps = 0
    k = 6
    laplacian_regularization = 0
    laplacian_normalization = 'rw'
    c_l = 1
    c_u = 1

    # number of labeled (unmasked) nodes provided to the hfs algorithm
    l = 4

    # mask labels
    Y_masked = mask_labels(Y, l)
    X,Y_masked,Y = sorting(X,Y_masked,Y)

    """
    compute hfs solution using either soft_hfs or hard_hfs
    """
    labels = hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    #labels = soft_hfs(X, Y_masked, c_l , c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    """
    Visualize results
    """
    plot_classification(X, Y, labels,  var=var, eps=0, k=k)
    accuracy = np.mean(labels == np.squeeze(Y))
    
    return accuracy

    
def soft_hfs(X, Y, c_l, c_u, laplacian_regularization, var=1, eps=0, k=0, laplacian_normalization=""):
    """
    Function to perform soft (unconstrained) HFS


    :param X: (n x m) matrix of m-dimensional samples
    :param Y: (n x 1) vector with nodes labels [1, ... , num_classes] (0 is unlabeled)
    :param c_l: coefficients for C matrix
    :param c_u: coefficients for C matrix
    :param laplacian_regularization:
    :param var:
    :param eps:
    :param k:
    :param laplacian_normalization:
    :return: labels, class assignments for each of the n nodes
    """

    num_samples = np.size(Y, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1
    
    C = np.zeros((num_samples,num_samples))
            
    L = build_laplacian_regularized(X,k=k)
    
    """
    Compute the target y for the linear system  
    y = (n x num_classes) target vector 
    l_idx = (l x num_classes) vector with indices of labeled nodes    
    u_idx = (u x num_classes) vector with indices of unlabeled nodes 
    """

    y = np.zeros((num_samples,num_classes))
    for i in range(num_samples):
        if Y[i] != 0:
            y[i][int(Y[i])-1]=1

    l_idx, u_idx = [], []
    for i in range(num_samples):
        if Y[i]==0:
            u_idx.append(i)
        else:
            l_idx.append(i)

    """
    compute the hfs solution, remember that you can use build_laplacian_regularized and build_similarity_graph
    f = (n x num_classes) hfs solution 
    C = (n x n) diagonal matrix with c_l for labeled samples and c_u otherwise    
    """
    
    for i in range(len(Y)):
        if Y[i]==0:
            C[i][i] = c_u
        else:
            C[i][i] = c_l

    
    inv = np.linalg.inv(L + C)    
    f = (y.T @ C @ inv).T
    
    """
    compute the labels assignment from the hfs solution 
    labels: (n x 1) class assignments [1, ... ,num_classes]  
    """
    labels = []
    for i in range(len(f)):
        lab = np.argmax([f[i][k] for k in range(num_classes)])+1
        labels.append(lab)
        
    return labels


def hard_vs_soft_hfs():
    """
    Function to compare hard and soft HFS.
    """
    # load the data
    in_data = loadmat(os.path.join('data', 'data_2moons_hfs.mat'))
    X = in_data['X']
    Y = in_data['Y'].squeeze()

    # automatically infer number of labels from samples
    num_samples = np.size(Y, 0)
    Cl = np.unique(Y)
    num_classes = len(Cl)-1
    
    # randomly sample 20 labels
    l = 20
    # mask labels
    Y_masked = mask_labels(Y, l)

    # Create some noisy labels
    Y_masked[Y_masked != 0] = label_noise(Y_masked[Y_masked != 0], 4)
    
    X,Y_masked,Y = sorting(X,Y_masked,Y)


    """
    choose parameters
    """
    var = 1
    eps = 0
    k = 6
    laplacian_regularization = 0
    laplacian_normalization = 'rw'
    c_l = 0.95
    c_u = 0.05

    """
    Compute hfs solution using soft_hfs() and hard_hfs().
    Remember to use Y_masked (the vector with some labels hidden as input and NOT Y (the vector with all labels 
    revealed)
    """
    hard_labels = hard_hfs(X, Y_masked, laplacian_regularization, var, eps, k, laplacian_normalization)
    soft_labels = soft_hfs(X, Y_masked, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)

    plot_classification_comparison(X, Y, hard_labels, soft_labels, var=var, eps=eps, k=k)
    accuracy = [np.mean(hard_labels == np.squeeze(Y)), np.mean(soft_labels == np.squeeze(Y))]
    return accuracy


if __name__ == '__main__':
    #accuracy = two_moons_hfs()
    accuracy = hard_vs_soft_hfs()
    print(accuracy)
    

