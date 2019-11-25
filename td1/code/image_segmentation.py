from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
import scipy.spatial.distance as sd

from build_similarity_graph import build_similarity_graph
from spectral_clustering import build_laplacian, spectral_clustering, choose_eigenvalues


def image_segmentation(input_img='fruit_salad.bmp',eig_max=15):
    """
    TO BE COMPLETED

    Function to perform image segmentation.

    :param input_img: name of the image file in /data (e.g. 'four_elements.bmp')
    """
    filename = os.path.join('data', input_img)

    X = io.imread(filename)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    #print(X.shape)
    
    im_side = np.size(X, 1)
    Xr = X.reshape(im_side ** 2, 3)
    #print(Xr.shape)
    
    """
    Y_rec should contain an index from 0 to c-1 where c is the     
     number of segments you want to split the image into          
    """

    """
    Choose parameters
    """
    
    var = 5
    k = 45
    laplacian_normalization = 'unn'

    W = build_similarity_graph(Xr, var=var, k=k)
        
    L = build_laplacian(W, laplacian_normalization)
    
    E,U = np.linalg.eig(L)
    indexes = np.argsort(E)
    E = E[indexes]
    U = U[:,indexes]
    chosen_eig_indices = choose_eigenvalues(E,eig_max)
    #chosen_eig_indices = [0,1,2,3]
    
    num_classes = len(chosen_eig_indices)

    #print(len(chosen_eig_indices))
        
    Y_rec = spectral_clustering(L, chosen_eig_indices, num_classes=num_classes)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.imshow(X)

    plt.subplot(1, 2, 2)
    Y_rec = Y_rec.reshape(im_side, im_side)
    plt.imshow(Y_rec)

    plt.show()


if __name__ == '__main__':
    image_segmentation(input_img='fruit_salad.bmp',eig_max=15)
    image_segmentation(input_img='four_elements.bmp',eig_max=15)
