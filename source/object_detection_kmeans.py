import numpy as np
import cv2
from collections import deque
from numpy.random import random

NUMBER_OF_CLUSTERS = 4
DISTANCE_THRESHOLD = 20
old_mean = (0, 0)

coords_history = deque([])
accel_history = deque([])


# Fill histories for rotation + randomize to avoid instant freeze
def init_history():
    for _ in range(30):
        coords_history.append((round(random() * 100), round(random() * 100)))
    for _ in range(10):
        accel_history.append(random())


init_history()


# Prepare incoming frames for analysis

def preprocess(image):
    image = cv2.findNonZero(255 - image)

    if image is not None:
        # Taken from https://docs.opencv.org/master/d1/d5c/tutorial_py_kmeans_opencv.html (Step 3)
        Z2 = np.squeeze(image, axis=1)
        Z2 = Z2.reshape((-1, 2))
        return np.float32(Z2)
    else:
        return None


# Calculate the average change in location
def calc_delta_avg():
    x, y = 0, 0

    # Sum coordinates history
    for i in range(len(coords_history)):
        if i > 1 and coords_history[i - 1][0] != 0 and coords_history[i - 1][1] != 0:
            x += (coords_history[i][0] - coords_history[i - 1][0]) / coords_history[i - 1][0]
            y += (coords_history[i][1] - coords_history[i - 1][1]) / coords_history[i - 1][1]

    # Calculate Euclidean distance for acceleration
    mean = np.sqrt(pow(x/30, 2) + pow(y/30, 2))

    accel_history.rotate(1)
    accel_history[0] = mean
    return mean


# Detect movement over time
def detect_movement():
    t = 0
    for i in accel_history:
        t += i
    
    # Calculate the average acceleration and compare to threshold
    return t / len(accel_history) < 0.004


# Perform the main image processing and analysis tasks
def process_frame(image):
    # Do data preprocessing
    Z2 = preprocess(image)

    if Z2 is not None:
        # Perform the kMeans cluster analysis
        # Using code from https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, _, center = cv2.kmeans(Z2, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Calculate average speed over frames
        coords_history.rotate(1)
        coords_history[0] = center[0]

        return center
    else:
        return []
