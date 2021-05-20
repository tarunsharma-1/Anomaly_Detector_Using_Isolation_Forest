
"""Author: Tarun Sharma

Project: Anomaly Detector

Purpose: Anomaly detector model training script

Original file is located at
    https://colab.research.google.com/drive/1MWfRhy2E-rZFXjbQbLJrNNM7oW2d1I6W
"""

from helper_fcn import load_dataset
from sklearn.ensemble import IsolationForest
import argparse
import pickle
# constructing the argument parser and parsing the arguments
#used for giving command line arguments, here we can give location of our dataset and output model
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output anomaly detection model")
args = vars(ap.parse_args())

# load and quantify our image dataset
print("[INFO] preparing dataset...")
data = load_dataset(args["dataset"], bins=(3, 3, 3))
# train the anomaly detection model using isolation forest
print("[INFO] fitting anomaly detection model...")
model = IsolationForest(n_estimators=100, contamination=0.01,
	random_state=42)
model.fit(data)

# serialize the anomaly detection model to disk
#we are using pickle here for output model
f = open(args["model"], "wb")
f.write(pickle.dumps(model))
f.close()
