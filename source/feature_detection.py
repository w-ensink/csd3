import numpy as np
import cv2

cap = cv2.VideoCapture('../assets/WIN_20210918_10_58_50_Pro.mp4')
img = cv2.imread('../assets/whitepixeltest.PNG')


def process_image(frame):
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    threshold = cv2.threshold(src=grayscale_image, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)[1]

    kernel = np.ones(shape=(5, 5), dtype=np.uint8)

    threshold = cv2.morphologyEx(src=threshold, op=cv2.MORPH_CLOSE, kernel=kernel)

    contours = cv2.findContours(image=threshold, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    thresh_copy = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        e = 0.005 * perimeter  # The bigger the fraction, the more sides are chopped off the original polygon
        contour = cv2.approxPolyDP(contour, epsilon=e, closed=True)
        cv2.drawContours(thresh_copy, [contour], contourIdx=-1, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(thresh_copy, [contour], (255, 0, 0))

    return thresh_copy


def main():
    while 1:
        ret, frame = cap.read()

        thresh_copy = process_image(frame)

        cv2.imshow('frame', frame)
        cv2.imshow('simplified contours', thresh_copy)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
