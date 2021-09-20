import numpy as np
import cv2

# feature detection
# - background subtraction
# - background = black, moving object = white
# - count black & white pixels to set percentage
# - canny edge detection to detect edge coordinates in realtime

cap = cv2.VideoCapture('../assets/WIN_20210918_10_58_50_Pro.mp4')
background_subtraction = cv2.createBackgroundSubtractorKNN()

while 1:
    ret, frame = cap.read()
    kernel = np.ones((5, 5), np.uint8)

    foreground_mask = background_subtraction.apply(frame)
    closing = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    # moving object
    white_pixels = cv2.countNonZero(closing)
    # print("white pixels: ", white_pixels)

    # background
    black_pixels = np.sum(closing == 0)
    # print("black pixels: ", black_pixels)

    percentage_white_pixels = (white_pixels / black_pixels) * 100
    # print("percentage of white pixels: ", percentage_white_pixels, "\n")

    # canny edge detection, coordinates variable contains all x and y values of the edge points
    edges = cv2.Canny(closing, 100, 150)
    indices = np.where(edges != [0])
    coordinates = list(zip(indices[0], indices[1]))
    print(coordinates)

    cv2.imshow('frame', frame)
    cv2.imshow('canny', edges)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()