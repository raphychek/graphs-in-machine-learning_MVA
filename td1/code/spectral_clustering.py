import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skm
import scipy
import scipy.spatial.distance as sd


from utils import plot_clustering_result, plot_the_bend, min_span_tree, plot_graph_matrix
from build_similarity_graph import build_similarity_graph
from generate_data import blobs, two_moons, point_and_circle


def build_laplacian(W, laplacian_normalization=""):
    """
    Compute graph Laplacian.

    :param W: adjacency matrix
    :param laplacian_normalization:  string selecting which version of the laplacian matrix to construct
                                     'unn':  unnormalized,
                                     'sym': symmetric normalization
                                     'rw':  random-walk normalization
    :return: L: (n x n) dimensional matrix representing the Laplacian of the graph
    """
    D = np.diag(np.sum(W,axis=1))
    L = D - W
    if laplacian_normalization=="unn":
        return L
    elif laplacian_normalization=="rw":
        D = np.linalg.inv(D)
        return np.eye(W.shape[0]) - D.dot(W)
    elif laplacian_normalization=="sym":
        D = np.linalg.inv(D)
        return np.eye(W.shape[0]) - ((np.sqrt(D)).dot(W)).dot((np.sqrt(D)))
    else:
        print("The parameter of laplacian_normalization should be chosen between 'unn', 'sym' or 'rw'")
        return L

def spectral_clustering(L, chosen_eig_indices, num_classes=2):
    """
    :param L: Graph Laplacian (standard or normalized)
    :param chosen_eig_indices: indices of eigenvectors to use for clustering
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.sparse.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    # eigenvalues and eigenvectors
    # sort these based on the eigenvalues    

    E,U = np.linalg.eig(L)
    indexes = np.argsort(E)
    E = E[indexes]
    U = U[:,indexes]

    """
    compute the clustering assignment from the eigenvectors        
    Y = (n x 1) cluster assignments [0,1,...,c-1]                   
    """
    eigen_vectors = U.real[:,chosen_eig_indices].copy()
    kmeans = KMeans(n_clusters=num_classes)
    Y = kmeans.fit(eigen_vectors).labels_
    
    return Y


def two_blobs_clustering():
    """
    TO BE COMPLETED

    Clustering of two blobs. Used in questions 2.1 and 2.2
    """

    # Get data and compute number of classes
    X, Y = blobs(50, n_blobs=2, blob_var=0.15, surplus=0)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    chosen_eig_indices = [0,1,2]    # indices of the ordered eigenvalues to pick
    
    
    if k==0: # compute epsilon
        dists = sd.cdist(X,X,'euclidean')  # dists[i, j] = euclidean distance between x_i and x_j

        min_tree = min_span_tree(dists)
        
        l = []
        n1,m1 = min_tree.shape
        
        for i in range(n1):
            for j in range(m1):
                if min_tree[i][j]==True:
                    l.append(dists[i][j])
    
        #distance_threshold = sorted(l)[-1]
        distance_threshold = sorted(l)[-2]
        
        eps = np.exp(-(distance_threshold)**2.0/(2*var))
    #####

    # build laplacian
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    plot_graph_matrix(X, Y, W)

    L = build_laplacian(W, laplacian_normalization)

    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    # Plot results
    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))
    
#two_blobs_clustering()


def choose_eigenvalues(eigenvalues,eig_max):
    """
    Function to choose the indices of which eigenvalues to use for clustering.

    :param eigenvalues: sorted eigenvalues (in ascending order)
    :return: indices of the eigenvalues to use
    """
    eig_ind = [0,1,2]
    for i in range(3,eig_max):
        eig_ind.append(i)
        if 2 * eigenvalues[i] < eigenvalues[i-1] + eigenvalues[i+1]:
            #print(i)
            break
    return eig_ind


def spectral_clustering_adaptive(L, num_classes=2, eig_max = 0):
    """
    Spectral clustering that adaptively chooses which eigenvalues to use.
    :param L: Graph Laplacian (standard or normalized)
    :param num_classes: number of clusters to compute (defaults to 2)
    :return: Y: Cluster assignments
    """

    """
    Use the function scipy.linalg.eig or the function scipy.linalg.eigs to compute:
    U = (n x n) eigenvector matrix           (sorted)
    E = (n x n) eigenvalue diagonal matrix   (sorted)
    """
    E,U = np.linalg.eig(L)
    indexes = np.argsort(E)
    E = E[indexes]
    U = U[:,indexes]

    eigen_indexes = choose_eigenvalues(E,eig_max)
    
    """
    compute the clustering assignment from the eigenvectors   
    Y = (n x 1) cluster assignments [1,2,...,c]                   
    """
    
    eigen_vectors = U.real[:,eigen_indexes].copy()
    kmeans = KMeans(n_clusters=num_classes)
    Y = kmeans.fit(eigen_vectors).labels_
    return Y


def find_the_bend(eig_max=15, blob_var=0.03):
    """
    TO BE COMPLETED

    Used in question 2.3
    :return:
    """
    eig_max -= 1 # to count starting from 0
    # the number of samples to generate
    num_samples = 600

    # Generate blobs and compute number of clusters
    X, Y = blobs(num_samples, 4, blob_var)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    var = 1.0     # exponential_euclidean's sigma^2
    laplacian_normalization = 'sym'  # either 'unn'normalized, 'sym'metric normalization or 'rw' random-walk normalization

    if k==0: # compute epsilon
        dists = sd.cdist(X,X,'euclidean')  # dists[i, j] = euclidean distance between x_i and x_j

        min_tree = min_span_tree(dists)
        
        l = []
        n1,m1 = min_tree.shape
        
        for i in range(n1):
            for j in range(m1):
                if min_tree[i][j]==True:
                    l.append(dists[i][j])
    
        #distance_threshold = sorted(l)[-1]
        distance_threshold = sorted(l)[-num_classes]
        
        eps = np.exp(-(distance_threshold)**2.0/(2*var))
    
    # build laplacian
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    L = build_laplacian(W, laplacian_normalization)

    """
    compute first 15 eigenvalues and call choose_eigenvalues() to choose which ones to use. 
    """
    eigenvalues,U = np.linalg.eig(L)
    indexes = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[indexes]
    U = U[:,indexes]
    
    chosen_eig_indices = choose_eigenvalues(eigenvalues, eig_max = eig_max)  # indices of the ordered eigenvalues to pick
    
    plt.plot(eigenvalues,[i for i in range(len(eigenvalues))],'r+')

    """
    compute spectral clustering solution using a non-adaptive method first, and an adaptive one after (see handout) 
    Y_rec = (n x 1) cluster assignments [0,1,..., c-1]    
    """
    # run spectral clustering
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)
    Y_rec_adaptive = spectral_clustering_adaptive(L, num_classes=num_classes, eig_max = eig_max)

    plot_the_bend(X, Y, L, Y_rec_adaptive, eigenvalues)

#find_the_bend(blob_var=0.20)

def two_moons_clustering(eig_max=15):
    """
    TO BE COMPLETED.

    Used in question 2.7
    """
    # Generate data and compute number of clusters
    X, Y = two_moons(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    var = 1.0  # exponential_euclidean's sigma^2

    laplacian_normalization = 'unn'
    
#    chosen_eig_indices = [0, 1, 2]    # indices of the ordered eigenvalues to pick

    if k==0: # compute epsilon
        dists = sd.cdist(X,X,'euclidean')  # dists[i, j] = euclidean distance between x_i and x_j

        min_tree = min_span_tree(dists)
        
        l = []
        n1,m1 = min_tree.shape
        
        for i in range(n1):
            for j in range(m1):
                if min_tree[i][j]==True:
                    l.append(dists[i][j])
    
        #distance_threshold = sorted(l)[-1]
        distance_threshold = sorted(l)[-1]
        
        eps = np.exp(-(distance_threshold)**2.0/(2*var))

    # build laplacian
    W = build_similarity_graph(X, var=var, eps=eps, k=k)
    L = build_laplacian(W, laplacian_normalization)
    
    # chose the eigenvalues
    eigenvalues,U = np.linalg.eig(L)
    indexes = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[indexes]
    U = U[:,indexes]
    chosen_eig_indices = choose_eigenvalues(eigenvalues, eig_max = eig_max)
    
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    plot_clustering_result(X, Y, L, Y_rec, KMeans(num_classes).fit_predict(X))

#two_moons_clustering()

def point_and_circle_clustering(eig_max=15):
    """
    TO BE COMPLETED.

    Used in question 2.8
    """
    # Generate data and compute number of clusters
    X, Y = point_and_circle(600)
    num_classes = len(np.unique(Y))

    """
    Choose parameters
    """
    k = 0
    var = 1.0  # exponential_euclidean's sigma^2

    #chosen_eig_indices = [1, 2, 3]    # indices of the ordered eigenvalues to pick

    if k==0: # compute epsilon
        dists = sd.cdist(X,X,'euclidean')  # dists[i, j] = euclidean distance between x_i and x_j

        min_tree = min_span_tree(dists)
        
        l = []
        n1,m1 = min_tree.shape
        
        for i in range(n1):
            for j in range(m1):
                if min_tree[i][j]==True:
                    l.append(dists[i][j])
    
        #distance_threshold = sorted(l)[-1]
        distance_threshold = sorted(l)[-1]
        
        eps = np.exp(-(distance_threshold)**2.0/(2*var))
        W = build_similarity_graph(X, var=var, eps=eps, k=k)

    # build laplacian
    else:
            W = build_similarity_graph(X, var=var, k=k)
    L_unn = build_laplacian(W, 'unn')
    L_norm = build_laplacian(W, 'sym')
    
    #eigenvalues,U = np.linalg.eig(L_unn)
    #indexes = np.argsort(eigenvalues)
    #eigenvalues = eigenvalues[indexes]
    #U = U[:,indexes]
    #chosen_eig_indices = choose_eigenvalues(eigenvalues, eig_max = eig_max)
    chosen_eig_indices = [0,1]
    
    Y_unn = spectral_clustering(L_unn, chosen_eig_indices, num_classes=num_classes)
    Y_norm = spectral_clustering(L_norm, chosen_eig_indices, num_classes=num_classes)

    plot_clustering_result(X, Y, L_unn, Y_unn, Y_norm, 1)

#point_and_circle_clustering()

def parameter_sensitivity(eig_max=15):
    """
    TO BE COMPLETED.

    A function to test spectral clustering sensitivity to parameter choice.

    Used in question 2.9
    """
    # the number of samples to generate
    num_samples = 500

    """
    Choose parameters
    """
    var = 1.0  # exponential_euclidean's sigma^2
    laplacian_normalization = 'unn'
    #chosen_eig_indices = [0, 1, 2]

    """
    Choose candidate parameters
    """
    parameter_candidate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # the number of neighbours for the graph or the epsilon threshold
    parameter_performance = []

    for k in parameter_candidate:
        # Generate data
        X, Y = two_moons(num_samples, 1, 0.02)
        num_classes = len(np.unique(Y))
        
        if k==0: # compute epsilon
            dists = sd.cdist(X,X,'euclidean')  # dists[i, j] = euclidean distance between x_i and x_j
    
            min_tree = min_span_tree(dists)
            
            l = []
            n1,m1 = min_tree.shape
            
            for i in range(n1):
                for j in range(m1):
                    if min_tree[i][j]==True:
                        l.append(dists[i][j])        
            distance_threshold = sorted(l)[-1]
            eps = np.exp(-(distance_threshold)**2.0/(2*var))
            W = build_similarity_graph(X, var=var, eps=eps, k=k)
        else:
            W = build_similarity_graph(X, k=k)
        L = build_laplacian(W, laplacian_normalization)
        
        eigenvalues,U = np.linalg.eig(L)
        indexes = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[indexes]
        U = U[:,indexes]
        chosen_eig_indices = choose_eigenvalues(eigenvalues, eig_max = eig_max)

        Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes)

        parameter_performance += [skm.adjusted_rand_score(Y, Y_rec)]

    plt.figure()
    plt.plot(parameter_candidate, parameter_performance)
    plt.title('parameter sensitivity')
    plt.show()

#parameter_sensitivity()