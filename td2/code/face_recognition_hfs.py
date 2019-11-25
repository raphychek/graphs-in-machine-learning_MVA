import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
import cv2
import os

from helper import *

from harmonic_function_solution import *


def offline_face_recognition():
    """
    Function to test offline face recognition.
    """

    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    frame_size = 96
    # Loading images
    images = np.zeros((100, frame_size ** 2))
    labels = np.zeros(100)

    for i in np.arange(10):
        for j in np.arange(10):
            im = imread("data/10faces/%d/%02d.jpg" % (i, j + 1))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            
            
            """
            I tried all the combinations of the following pretreatments, after finetuning each one individually
            """
            #gray_face = cv2.equalizeHist(gray_face)
            
            #gray_face = cv2.GaussianBlur(gray_face, (43, 43), 10) # accuracy of 0.84 with equalize, 0.86 without
            #gray_face = cv2.bilateralFilter(gray_face,10,100,100) # accuracy of 0.79 with equalize, 0.83 without
            gray_face = cv2.blur(gray_face,(50,50)) # accuracy of 0.83 with equalize, 0.87 without
            #gray_face = cv2.blur(gray_face,(60,60)) # only test working with Hard-HFS
            
            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf
            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False

    if plot_the_dataset:
        plt.figure(1)
        for i in range(100):
            plt.subplot(10, 10, i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size))
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person '+r)
        plt.show()

    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    
    mlabels = labels.copy()
    for i in range(10):
        mask = np.arange(10)
        np.random.shuffle(mask)
        mask = mask[:6]
        for m in mask:
            mlabels[m * 10 + i] = 0
    print(mlabels)

    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = 100
    var = 10000
    eps = 0
    k = 6
    laplacian_regularization = gamma
    laplacian_normalization = ""
    c_l = 0.95
    c_u = 0.05

    images,mlabels,labels = sorting(images,mlabels,labels)
    # hard or soft HFS
    #rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)
    rlabels = hard_hfs(images, mlabels, laplacian_regularization, var, eps, k, laplacian_normalization)


    # Plots #
    plt.subplot(121)
    plt.imshow(labels.reshape((10, 10)))

    plt.subplot(122)
    plt.imshow(np.array(rlabels).reshape((10, 10)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()

    
def offline_face_recognition_augmented():
    # Parameters
    cc = cv2.CascadeClassifier(os.path.join('data', 'haarcascade_frontalface_default.xml'))
    frame_size = 96
    gamma = .95
    nbimgs = 50
    # Loading images
    images = np.zeros((10 * nbimgs, frame_size ** 2))
    labels = np.zeros(10 * nbimgs)
    var=10000
    
    sizes = []

    for i in np.arange(10):
        imgdir = "data/extended_dataset/%d" % i
        imgfns = os.listdir(imgdir)
        sizes.append(len(imgfns))
        for j, imgfn in enumerate(np.random.choice(imgfns, size=nbimgs)):
            im = imread("{}/{}".format(imgdir, imgfn))
            box = cc.detectMultiScale(im)
            top_face = {"area": 0, "box": (0, 0, *im.shape[:2])}

            for cfx, cfy, clx, cly in box:
                face_area = clx * cly
                if face_area > top_face["area"]:
                    top_face["area"] = face_area
                    top_face["box"] = [cfx, cfy, clx, cly]

            fx, fy, lx, ly = top_face["box"]
            gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            gray_face = gray_im[fy:fy + ly, fx:fx + lx]

            """
            Apply preprocessing to balance the image (color/lightning), such as filtering 
            (cv.boxFilter, cv.GaussianBlur, cv.bilinearFilter) and
            equalization (cv.equalizeHist).   
            """
            #gray_face = cv2.equalizeHist(gray_face)
            #gray_face = cv2.GaussianBlur(gray_face, (43, 43), 10)
            gray_face = cv2.blur(gray_face,(50,50))
            
            gf = cv2.resize(gray_face, (frame_size, frame_size)).astype(np.float)
            gf -= gf.mean()
            gf /= gf.max()
            gray_face = gf
            # resize the face and reshape it to a row vector, record labels
            images[j * 10 + i] = gray_face.reshape((-1))
            labels[j * 10 + i] = i + 1

    """
     If you want to plot the dataset, set the following variable to True
    """
    plot_the_dataset = False
    if plot_the_dataset:

        plt.figure(1)
        for i in range(10 * nbimgs):
            plt.subplot(nbimgs,10,i+1)
            plt.axis('off')
            plt.imshow(images[i].reshape(frame_size, frame_size))
            r='{:d}'.format(i+1)
            if i < 10:
                plt.title('Person '+r)
        plt.show()

    """
    select 4 random labels per person and reveal them  
    Y_masked: (n x 1) masked label vector, where entries Y_i take a values in [1, ..., num_classes] if the node is  
              labeled, or 0 if the node is unlabeled (masked)   
    """
    
    mlabels = labels.copy()
    for i in range(10):
        mask = np.arange(50)
        np.random.shuffle(mask)
        mask = mask[:6]
        for m in mask:
            mlabels[m * 10 + i] = 0
    

    """
     Choose the experiment parameter and compute hfs solution using either soft_hfs or hard_hfs  
    """
    gamma = 100
    var = 10000
    eps = 0
    k = 6
    laplacian_regularization = gamma
    laplacian_normalization = ""
    c_l = 0.95
    c_u = 0.05

    images,mlabels,labels = sorting(images,mlabels,labels)
    # hard or soft HFS
    rlabels = soft_hfs(images, mlabels, c_l, c_u, laplacian_regularization, var, eps, k, laplacian_normalization)
    #rlabels = hard_hfs(images, mlabels, laplacian_regularization, var, eps, k, laplacian_normalization)


    """
    Plots
    """
    plt.subplot(121)
    plt.imshow(labels.reshape((-1, 10)))

    plt.subplot(122)
    plt.imshow(np.array(rlabels).reshape((-1, 10)))
    plt.title("Acc: {}".format(np.equal(rlabels, labels).mean()))

    plt.show()


if __name__ == '__main__':
    offline_face_recognition()
    #offline_face_recognition_augmented()

