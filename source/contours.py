# inspired by: https://automaticaddison.com/how-to-draw-contours-around-objects-using-opencv/

import cv2
import numpy as np


def create_blank_image():
    return np.ones(shape=(450, 600, 3), dtype=np.uint8)


def get_screen_contour(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to black and white.
    # Modify the threshold (e.g. 75 for tshirt.jpg) accordingly depending on how to output looks.
    # If you have a dark item on a light background, use cv2.THRESH_BINARY_INV and consider
    # changing the lower color threshold to 115.
    threshold = cv2.threshold(src=grayscale_image, thresh=160, maxval=255, type=cv2.THRESH_BINARY)[1]

    kernel = np.ones(shape=(5, 5), dtype=np.uint8)

    # Use the kernel to perform morphological opening
    threshold = cv2.morphologyEx(src=threshold, op=cv2.MORPH_OPEN, kernel=kernel)

    # Find the contours
    contours = cv2.findContours(image=threshold, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    blank_image = create_blank_image()
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]


def get_corners(contour):
    peri = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.04 * peri, True)
    return [tuple(c[0]) for c in corners]


def main():
    img = cv2.imread('../assets/screen_transform_test.jpg')

    corners = get_corners(get_screen_contour(img))

    cv2.line(img, corners[0], corners[1], (0, 255, 0), 10)
    cv2.line(img, corners[1], corners[2], (0, 255, 0), 10)
    cv2.line(img, corners[2], corners[3], (0, 255, 0), 10)
    cv2.line(img, corners[3], corners[0], (0, 255, 0), 10)

    cv2.imshow('image', img)

    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
