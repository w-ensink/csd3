import numpy as np
import cv2
from collections import deque
import draw

# feature detection
# - background subtraction
# - background = black, moving object = white
# - count black & white pixels to set percentage

# cap = cv2.VideoCapture(2)
from numpy.random import random

background_subtraction = cv2.createBackgroundSubtractorKNN()
cap = cv2.VideoCapture(2)

running = True
w = 1366
h = 768
freeze_frame = None
frozen = False

coords_history = deque([])
accel_history = deque([])


# Fill histories for rotation + randomize to avoid instant freeze
def init_history():
    for _ in range(30):
        coords_history.append((round(random() * 100), round(random() * 100)))
    for _ in range(10):
        accel_history.append(random())


init_history()


def bg_subtract(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (w, h))

    # Run masking process
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = background_subtraction.apply(img)
    return cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)


# Prepare incoming frames for analysis
def preprocess(image):
    c = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    Z2 = cv2.findNonZero(image)

    if Z2 is not None:
        Z2 = np.squeeze(Z2, axis=1)
        Z2 = Z2.reshape((-1, 2))
        return c, np.float32(Z2)
    else:
        return c, None


# Calculate the average change in location
def calc_delta_avg():
    x = 0
    y = 0
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
def process_frame(image, returnval="image"):
    global freeze_frame, frozen
    # Do data preprocessing
    img = bg_subtract(image)
    c, Z2 = preprocess(img)

    rtn = image
    center = []

    if Z2 is not None:
        # Perform the kMeans cluster analysis
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret2, label, center = cv2.kmeans(Z2, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Calculate average speed over frames
        coords_history.rotate(1)
        coords_history[0] = center[0]
        speed = calc_delta_avg()

        # Detect movement based on calculated speed
        freeze = detect_movement()
        if returnval == "image":
            rtn = draw.centers(center, c, (w, h), speed, freeze)

        if freeze:
            freeze_frame = rtn
            frozen = True

    if returnval == "image":
        return rtn
    elif returnval == "points":
        return center


# Main program loop
while False:
    # Get and resize
    ret, frame = cap.read()
    img2 = bg_subtract(frame)

    # Perform image processing
    if not frozen:
        ol = process_frame(img2)
        do_frame_analysis()
    else:
        ol = freeze_frame
    # ============================

    # Display the window
    cv2.imshow('detect', ol)

    # Handle keyboard input
    k = cv2.waitKey(30) & 0xff
    if type == "image" or k == 27:
        running = False
    elif k == 32:
        frozen = False
        init_history()

cap.release()
cv2.destroyAllWindows()
