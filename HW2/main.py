import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from sklearn import cluster
from sklearn.datasets.samples_generator import make_blobs


def doKmeansCluster(image, numberOfClusters=3):
    x, y, z = img.shape
    image_2d = img.reshape(x*y, z)
    plt.imshow(image_2d)
    plt.show()
    print('Clustering the image using %d clusters' % (numberOfClusters))
    kmeans_cluster = cluster.KMeans(n_clusters=numberOfClusters)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    resultImage = cluster_centers[cluster_labels].reshape(x, y, z)
    return resultImage.astype(int)


def doMeanShift(image):
    image_2d = np.reshape(image, [-1, 3])
    bandwidth = cluster.estimate_bandwidth(
        image_2d, quantile=0.1, n_samples=100)
    print(bandwidth)
    meanShift_cluster = cluster.MeanShift(
        bandwidth=bandwidth, bin_seeding=True)
    meanShift_cluster.fit(image_2d)
    cluster_centers = meanShift_cluster.cluster_centers_
    cluster_labels = meanShift_cluster.labels_
    print("number of predicted clusters : %d" % len(np.unique(cluster_labels)))
    segmented_image = np.reshape(cluster_labels, image.shape[:2])
    return segmented_image.astype(int)


print('Loading image')
img = mpimg.imread('SunnyLake.bmp')

print('Clustering using KMeans algorithim')
plt.subplot(231), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(
    doKmeansCluster(img, 3)), plt.title('Clustered with K=3')
plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(
    doKmeansCluster(img, 5)), plt.title('Clustered with K=5')
plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(
    doKmeansCluster(img, 7)), plt.title('Clustered with K=7')
plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(
    doKmeansCluster(img, 9)), plt.title('Clustered with K=9')
plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(
    doKmeansCluster(img, 12)), plt.title('Clustered with K=12')
plt.xticks([]), plt.yticks([])
plt.show()

print('Clustering using MeanShift algorithim')
plt.xticks([]), plt.yticks([])
plt.subplot(121), plt.imshow(
    img), plt.title('')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(
    doMeanShift(img)), plt.title('')
plt.xticks([]), plt.yticks([])
plt.show()
