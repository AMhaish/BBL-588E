import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.333, 0.333, 0.333])


def gaussianNoise(image, mean, variance):
    row, col, ch = image.shape
    mean = 0
    sigma = variance**0.5
    noisy = image + np.random.normal(mean, sigma, img.shape)
    noisy_img_clipped = np.clip(noisy, 0, 255)
    return noisy_img_clipped.astype(int)


print('Loading image')
img = mpimg.imread('SunnyLake.bmp')
print('Converting image to grayscale by taking the average values of R, G, B channels')
grayImage = rgb2gray(img)
plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(grayImage, cmap=plt.get_cmap(
    'gray')), plt.title('Grayscale version')
plt.xticks([]), plt.yticks([])
plt.show()
print('Getting the histogram of the gray scale image')
counts, bins = np.histogram(grayImage, range(255))
plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
plt.title('Image histogram')
plt.xlim([-0.5, 255.5])
plt.show()
print('Threshold the gray scale image')
threshold = 9
binImageCollection = cv2.threshold(
    grayImage, threshold, 255, cv2.THRESH_BINARY)[1]
plt.title('Thresholded image')
plt.imshow(binImageCollection, cmap=plt.get_cmap(
    'gray'))
plt.show()
print('Adding Gaussian noises to the orginal image')
guss1 = gaussianNoise(img, 0, 1)
guss2 = gaussianNoise(img, 0, 5)
guss3 = gaussianNoise(img, 0, 10)
guss4 = gaussianNoise(img, 0, 20)
plt.subplot(221), plt.imshow(guss1), plt.title(
    'Gaussian: 1 variance')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(guss2), plt.title(
    'Gaussian: 5 variance')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(guss3), plt.title(
    'Gaussian: 10 variance')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(guss4), plt.title(
    'Gaussian: 20 variance')
plt.xticks([]), plt.yticks([])
plt.show()
gussGrayImage1 = rgb2gray(guss1)
gussGrayImage2 = rgb2gray(guss2)
gussGrayImage3 = rgb2gray(guss3)
gussGrayImage4 = rgb2gray(guss4)
plt.subplot(221), plt.imshow(gussGrayImage1, cmap=plt.get_cmap(
    'gray')), plt.title(
    'Gaussian: 1 variance')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(gussGrayImage2, cmap=plt.get_cmap(
    'gray')), plt.title(
    'Gaussian: 5 variance')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(gussGrayImage3, cmap=plt.get_cmap(
    'gray')), plt.title(
    'Gaussian: 10 variance')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(gussGrayImage4, cmap=plt.get_cmap(
    'gray')), plt.title(
    'Gaussian: 20 variance')
plt.xticks([]), plt.yticks([])
plt.show()
print('Defining kernels for lowpass filters')
h1Kernel = np.ones((3, 3), np.float32)/9
h2Kernel = np.ones((5, 5), np.float32)/25
h3Kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)/16
print('Filtering the Gaussian variance 1 grayscale image with low pass filters')
filtered = cv2.filter2D(gussGrayImage1, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage1, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage1, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage1, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 1')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 5x5')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Filtering the Gaussian variance 5 grayscale image with low pass filters')
filtered = cv2.filter2D(gussGrayImage2, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage2, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage2, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage2, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 5')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 5x5')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Filtering the Gaussian variance 10 grayscale image with low pass filters')
filtered = cv2.filter2D(gussGrayImage3, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage3, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage3, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage3, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 10')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 5x5')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Filtering the Gaussian variance 20 grayscale image with low pass filters')
filtered = cv2.filter2D(gussGrayImage4, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage4, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage4, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage4, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 20')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Lowpass filter 5x5')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Defining kernels for highpass filters')
h1Kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)
h2Kernel = np.array(
    [[0.17, 0.67, 0.17], [0.67, -3.33, 0.67], [0.17, 0.67, 0.17]], np.float32)
h3Kernel = np.array([[-1, -1, -1], [-1, 12.5, -1], [-1, -1, -1]], np.float32)
print('Filter the Gaussian variance 1 grayscale image with high pass filters')
filtered = cv2.filter2D(gussGrayImage1, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage1, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage1, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage1, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 1')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Laplacian filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Highpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('High Boost filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Filter the Gaussian variance 5 grayscale image with high pass filters')
filtered = cv2.filter2D(gussGrayImage2, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage2, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage2, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage2, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 5')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Laplacian filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Highpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('High Boost filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Filter the Gaussian variance 10 grayscale image with high pass filters')
filtered = cv2.filter2D(gussGrayImage3, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage3, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage3, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage3, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 10')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Laplacian filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Highpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('High Boost filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Filter the Gaussian variance 20 grayscale image with high pass filters')
filtered = cv2.filter2D(gussGrayImage4, -1, h1Kernel)
filtered2 = cv2.filter2D(gussGrayImage4, -1, h2Kernel)
filtered3 = cv2.filter2D(gussGrayImage4, -1, h3Kernel)
plt.subplot(221), plt.imshow(gussGrayImage4, cmap=plt.get_cmap(
    'gray')), plt.title('Gaussian variance 20')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(filtered, cmap=plt.get_cmap(
    'gray')), plt.title('Laplacian filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(filtered2, cmap=plt.get_cmap(
    'gray')), plt.title('Highpass filter 3x3')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered3, cmap=plt.get_cmap(
    'gray')), plt.title('High Boost filter')
plt.xticks([]), plt.yticks([])
plt.show()
print('Filter noisy image with median filter')
noisyImg = mpimg.imread('Figure_1.png')
final = cv2.medianBlur(noisyImg, 3)
plt.subplot(121), plt.imshow(noisyImg), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(final), plt.title('After Median filter')
plt.xticks([]), plt.yticks([])
plt.show()
