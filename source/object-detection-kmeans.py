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
    for i in range(len(coords_history)):
        if i > 1 and coords_history[i - 1][0] != 0 and coords_history[i - 1][1] != 0:
            x += (coords_history[i][0] - coords_history[i - 1][0]) / coords_history[i - 1][0]
            y += (coords_history[i][1] - coords_history[i - 1][1]) / coords_history[i - 1][1]
    mean = np.sqrt(pow(x/30, 2) + pow(y/30, 2))

    accel_history.rotate(1)
    accel_history[0] = mean
    return mean


# Detect movement over time
def detect_movement():
    t = 0
    for i in accel_history:
        t += i
    return t / len(accel_history) < 0.004


def kMeans(image):
    global freeze_frame, frozen
    # convert to np.float32
    c, Z2 = preprocess(image)

    if Z2 is not None:
        # define criteria and apply kMeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret2, label, center = cv2.kmeans(Z2, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Calculate average speed per n frames
        coords_history.rotate(1)
        coords_history[0] = center[0]
        speed = calc_delta_avg()

        # Detect movement based on calculated speed
        freeze = detect_movement()
        if freeze:
            freeze_frame = draw.centers(center, c, (w, h), speed, freeze)
            frozen = True
            return freeze_frame

        return draw.centers(center, c, (w, h), speed, freeze)
    else:
        return image


# Paul's code
def do_frame_analysis():
    # moving object
    white_pixels = cv2.countNonZero(img)
    print("white pixels: ", white_pixels)

    # background
    black_pixels = np.sum(img == 0)
    print("black pixels: ", black_pixels)

    percentage_white_pixels = (white_pixels / black_pixels) * 100
    print("percentage of white pixels: ", percentage_white_pixels, "\n")


# Main program loop
while running:
    # Get and resize
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (w, h))

    # Run masking process
    kernel = np.ones((5, 5), np.uint8)
    foreground_mask = background_subtraction.apply(img)
    img2 = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    # Perform image processing
    if not frozen:
        ol = kMeans(img2)
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
