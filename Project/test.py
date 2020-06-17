# import the necessary packages
from pyimagesearch.features import quantify_image, load_dataset_with_paths
import argparse
import pickle
import cv2
import matplotlib.pyplot as plt
import numpy as np
# construct the argument parser and parse the arguments


def printImagesPathsAndPredictionResults(data, preds):
    for idx, val in enumerate(data):
        print("%s with the result: %d" % (val[0], preds[idx]))


def showImagesAndPredictionResults(data, preds):
    for idx, val in enumerate(data):
        label = "anomaly" if preds[idx] == -1 else "normal"
        color = (0, 0, 255) if preds[idx] == -1 else (0, 255, 0)
        # draw the predicted label text on the original image
        image = cv2.imread(val[0])
        cv2.putText(image, label, (10,  25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # display the image
        cv2.imshow("Output", image)
        cv2.waitKey(0)


def processDataset(title, data, sameType, model):
    clusteringInput = []
    for item in data:
        clusteringInput.append(item[1])
    # use the anomaly detector model and extracted features to determine
    # if the example image is an anomaly or not
    preds = model.predict(clusteringInput)
    succeededCounter = 0
    for idx, val in enumerate(preds):
        if ((sameType and preds[idx] == 1) or (not sameType and preds[idx] == -1)):
            succeededCounter += 1
    print("[INFO] %s finished with accuracy %f" %
          (title, succeededCounter / (len(preds))))
    #printImagesPathsAndPredictionResults(data, preds)
    #showImagesAndPredictionResults(data, preds)


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained anomaly detection model")
ap.add_argument("-d", "--training", required=True,
                help="path to test dataset of same scense")
ap.add_argument("-ds", "--same", required=True,
                help="path to test dataset of same scense")
ap.add_argument("-ddg", "--differentgreen", required=True,
                help="path to test dataset of different scenses with green color ranges")
ap.add_argument("-dd", "--different", required=True,
                help="path to test dataset of different scenses")
args = vars(ap.parse_args())
print("[INFO] preparing dataset...")
data = load_dataset_with_paths(args["training"], bins=(3, 3, 3))
data_s = load_dataset_with_paths(args["same"], bins=(3, 3, 3))
data_dg = load_dataset_with_paths(args["differentgreen"], bins=(3, 3, 3))
data_d = load_dataset_with_paths(args["different"], bins=(3, 3, 3))

# load the anomaly detection model
print("[INFO] loading anomaly detection model...")
model = pickle.loads(open(args["model"], "rb").read())

print("[INFO] processing training dataset")
processDataset("traning set", data, True, model)

print("[INFO] processing same scenses dataset")
processDataset("same scenses dataset", data_s, True, model)

print("[INFO] processing different scenses with green color ranges dataset")
processDataset("different scenses with green color ranges dataset",
               data_dg, False, model)

print("[INFO] processing different scenses dataset")
processDataset("different scenses dataset", data_d, False, model)
