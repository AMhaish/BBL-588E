import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn import cluster
from sklearn.datasets.samples_generator import make_blobs


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.333, 0.333, 0.333])


def doKmeansClusterOnRGB(img, numberOfClusters=3):
    x, y, z = img.shape
    algo_input = img.reshape(x*y, z)
    print('Clustering the image using %d clusters' % (numberOfClusters))
    kmeans_cluster = cluster.KMeans(n_clusters=numberOfClusters)
    kmeans_cluster.fit(algo_input)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    resultImage = cluster_centers[cluster_labels].reshape(x, y, z)
    return resultImage.astype(int)


def doKmeansClusterOnGrayscale(img, numberOfClusters=3):
    gimg = np.dot(img[..., :3], [0.333, 0.333, 0.333])
    x, y = gimg.shape
    algo_input = gimg.reshape(x*y, 1)
    print('Clustering the image using %d clusters' % (numberOfClusters))
    kmeans_cluster = cluster.KMeans(n_clusters=numberOfClusters)
    kmeans_cluster.fit(algo_input)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    resultImage = cluster_centers[cluster_labels].reshape(x, y)
    return resultImage.astype(int)


def doMeanShiftOnGray(image, bandwidth=None):
    gimg = np.dot(img[..., :3], [0.333, 0.333, 0.333])
    x, y = gimg.shape
    algo_input = gimg.reshape(x*y, 1)
    if bandwidth is None:
        bandwidth = cluster.estimate_bandwidth(
            algo_input, quantile=0.1, n_samples=100)
    meanShift_cluster = cluster.MeanShift(
        bandwidth=bandwidth, bin_seeding=True)
    meanShift_cluster.fit(algo_input)
    cluster_centers = meanShift_cluster.cluster_centers_
    cluster_labels = meanShift_cluster.labels_
    print("number of predicted clusters : %d" % len(np.unique(cluster_labels)))
    resultImage = cluster_centers[cluster_labels].reshape(x, y)
    return resultImage.astype(int)


def doMeanShiftOnRGB(image, bandwidth=None):
    x, y, z = img.shape
    algo_input = img.reshape(x*y, z)
    if bandwidth is None:
        bandwidth = cluster.estimate_bandwidth(
            algo_input, quantile=0.1, n_samples=100)
    meanShift_cluster = cluster.MeanShift(
        bandwidth=bandwidth, bin_seeding=True)
    meanShift_cluster.fit(algo_input)
    cluster_centers = meanShift_cluster.cluster_centers_
    cluster_labels = meanShift_cluster.labels_
    print("number of predicted clusters : %d" % len(np.unique(cluster_labels)))
    resultImage = cluster_centers[cluster_labels].reshape(x, y, z)
    return resultImage.astype(int)


print('Loading image')
img = mpimg.imread('SunnyLake.bmp')

print('Clustering using MeanShift algorithm')
plt.xticks([]), plt.yticks([])
plt.subplot(431), plt.imshow(
    img), plt.title('Original image')
plt.xticks([]), plt.yticks([])
plt.subplot(432), plt.imshow(
    doMeanShiftOnGray(img, 3), cmap=plt.get_cmap(
        'gray')), plt.title('MeanShift on Gray scale version with 3 bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(433), plt.imshow(
    doMeanShiftOnGray(img, 5), cmap=plt.get_cmap(
        'gray')), plt.title('MeanShift on Gray scale version with 5 bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(434), plt.imshow(
    doMeanShiftOnGray(img, 7), cmap=plt.get_cmap(
        'gray')), plt.title('MeanShift on Gray scale version with 7 bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(435), plt.imshow(
    doMeanShiftOnGray(img), cmap=plt.get_cmap(
        'gray')), plt.title('MeanShift on Gray scale version with auto bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(436), plt.imshow(
    doMeanShiftOnGray(img, 15), cmap=plt.get_cmap(
        'gray')), plt.title('MeanShift on Gray scale version with 15 bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(437), plt.imshow(
    img), plt.title('Original image')
plt.xticks([]), plt.yticks([])
plt.subplot(438), plt.imshow(
    doMeanShiftOnRGB(img, 10)), plt.title('MeanShift on RGB version with 10 bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(439), plt.imshow(
    doMeanShiftOnRGB(img, 20)), plt.title('MeanShift on RGB version with 20 bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 10), plt.imshow(
    doMeanShiftOnRGB(img)), plt.title('MeanShift on RGB version with auto bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11), plt.imshow(
    doMeanShiftOnRGB(img, 40)), plt.title('MeanShift on RGB version with 40 bandwidth')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 12), plt.imshow(
    doMeanShiftOnRGB(img, 50)), plt.title('MeanShift on RGB version with 50 bandwidth')
plt.xticks([]), plt.yticks([])
plt.show()


print('Clustering using KMeans algorithm')
plt.subplot(431), plt.imshow(img, cmap=plt.get_cmap(
    'gray')), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(432), plt.imshow(
    doKmeansClusterOnGrayscale(img, 3), cmap=plt.get_cmap(
        'gray')), plt.title('Clustered with K=3 on grayscale')
plt.xticks([]), plt.yticks([])
plt.subplot(433), plt.imshow(
    doKmeansClusterOnGrayscale(img, 5), cmap=plt.get_cmap(
        'gray')), plt.title('Clustered with K=5 on grayscale')
plt.xticks([]), plt.yticks([])
plt.subplot(434), plt.imshow(
    doKmeansClusterOnGrayscale(img, 7), cmap=plt.get_cmap(
        'gray')), plt.title('Clustered with K=7 on grayscale')
plt.xticks([]), plt.yticks([])
plt.subplot(435), plt.imshow(
    doKmeansClusterOnGrayscale(img, 9), cmap=plt.get_cmap(
        'gray')), plt.title('Clustered with K=9 on grayscale')
plt.xticks([]), plt.yticks([])
plt.subplot(436), plt.imshow(
    doKmeansClusterOnGrayscale(img, 12), cmap=plt.get_cmap(
        'gray')), plt.title('Clustered with K=12 on grayscale')
plt.xticks([]), plt.yticks([])
plt.subplot(437), plt.imshow(img, cmap=plt.get_cmap(
    'gray')), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(438), plt.imshow(
    doKmeansClusterOnRGB(img, 3)), plt.title('Clustered with K=3 on RGB featrue vector')
plt.xticks([]), plt.yticks([])
plt.subplot(439), plt.imshow(
    doKmeansClusterOnRGB(img, 5)), plt.title('Clustered with K=5 on featrue vector')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 10), plt.imshow(
    doKmeansClusterOnRGB(img, 7)), plt.title('Clustered with K=7 on featrue vector')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 11), plt.imshow(
    doKmeansClusterOnRGB(img, 9)), plt.title('Clustered with K=9 on featrue vector')
plt.xticks([]), plt.yticks([])
plt.subplot(4, 3, 12), plt.imshow(
    doKmeansClusterOnRGB(img, 12)), plt.title('Clustered with K=12 on featrue vector')
plt.xticks([]), plt.yticks([])
plt.show()
