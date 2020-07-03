# import the necessary packages
from pyimagesearch.features import load_dataset
from sklearn.ensemble import RandomForestClassifier
import argparse
import pickle
import time
start_time = time.time()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to dataset of images")
ap.add_argument("-m", "--model", required=True,
                help="path to output anomaly detection model")
args = vars(ap.parse_args())

# load and quantify our image dataset
print("[INFO] preparing dataset...")
data = load_dataset(args["dataset"], bins=(3, 3, 3))
print(data[1, 1])
# train the anomaly detection model
print("[INFO] fitting anomaly detection model...")
model = RandomForestClassifier(n_jobs=2, random_state=0)
model.fit(data)

# serialize the anomaly detection model to disk
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()
print("--- %s seconds ---" % (time.time() - start_time))
