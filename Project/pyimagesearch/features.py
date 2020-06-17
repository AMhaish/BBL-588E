from imutils import paths
import numpy as np
import cv2
import matplotlib.pyplot as plt


def quantify_image(image, bins=(4, 6, 3), draw=False):
    # compute a 3D color histogram over the image and normalize it
    hist = cv2.calcHist([image], [0, 1, 2], None,
                        bins, [0, 180, 0, 256, 0, 256])
    normHist = cv2.normalize(hist, hist).flatten()
    # if draw == True:
    #    color = ('b', 'g', 'r')
    #    for i, col in enumerate(color):
    #        plt.plot(hist[i], color=col)
    #        plt.xlim([0, 6])
    #    plt.show()
    # return the histogram
    return normHist


def load_dataset(datasetPath, bins):
    # grab the paths to all images in our dataset directory, then
    # initialize our lists of images
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image and convert it to the HSV color space
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # quantify the image and update the data list
        features = quantify_image(image, bins)
        data.append(features)
    # return our data list as a NumPy array
    return np.array(data)


def load_dataset_with_paths(datasetPath, bins):
    # grab the paths to all images in our dataset directory, then
    # initialize our lists of images
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image and convert it to the HSV color space
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # quantify the image and update the data list
        features = quantify_image(image, bins)
        data.append((imagePath, features))
    # return our data list as a NumPy array
    return data
