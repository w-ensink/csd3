import numpy as np
import cv2

# feature detection
# - background subtraction
# - background = black, moving object = white
# - count black & white pixels to set percentage

cap = cv2.VideoCapture('../assets/WIN_20210918_10_58_50_Pro.mp4')
background_subtraction = cv2.createBackgroundSubtractorKNN()

while 1:
    ret, frame = cap.read()
    kernel = np.ones((5, 5), np.uint8)

    foreground_mask = background_subtraction.apply(frame)
    closing = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('frame', frame)
    cv2.imshow('closing', closing)

    # moving object
    white_pixels = cv2.countNonZero(closing)
    print("white pixels: ", white_pixels)

    # background
    black_pixels = np.sum(closing == 0)
    print("black pixels: ", black_pixels)

    percentage_white_pixels = (white_pixels / black_pixels) * 100
    print("percentage of white pixels: ", percentage_white_pixels, "\n")

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
