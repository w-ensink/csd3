import numpy as np
import cv2
from object_detection_kmeans import process_contours
from utility import Frame, Features

cap = cv2.VideoCapture(0)
vid = cv2.VideoCapture('../assets/WIN_20210918_10_58_50_Pro.mp4')
img = cv2.imread('../assets/whitepixeltest.PNG')


def fill_frame(frame: Frame):
    frame_width = len(frame[0])
    frame_height = len(frame)

    left_upper = [0, 0]
    right_upper = [frame_width, 0]
    left_lower = [0, frame_height]
    right_lower = [frame_width, frame_height]

    frame_square = np.array([[left_upper, right_upper, right_lower, left_lower]], dtype=np.int32)

    cv2.fillPoly(frame, frame_square, (255, 255, 255))


def get_contours(frame: Frame):
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    threshold = cv2.threshold(src=grayscale_image, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones(shape=(5, 5), dtype=np.uint8)
    threshold = cv2.morphologyEx(src=threshold, op=cv2.MORPH_CLOSE, kernel=kernel)

    contours = cv2.findContours(image=threshold, mode=cv2.RETR_CCOMP, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    return contours


def get_center_points(contours):
    return process_contours(contours)


# TODO: get center point detection working
def detect_features(frame: Frame) -> Features:
    contours = get_contours(frame)
    center_points = get_center_points(contours)
    return Features(center_points, contours)


def draw_contour_coordinates_text(frame: Frame, contour: np.ndarray) -> Frame:
    n = contour.ravel()
    i = 0

    for _ in n:
        if i % 2 == 0:
            x = n[i]
            y = n[i + 1]
            text_color = (0, 255, 0)

            if i == 0:
                # text on topmost co-ordinate.
                cv2.putText(frame, "Arrow tip", (x, y), 0, 0.5, text_color)
            else:
                # text on remaining co-ordinates.
                cv2.putText(frame, f'{x} {y}', (x, y), 0, 0.5, text_color)
        i = i + 1
    return frame


def render_image(frame: Frame, features: Features) -> Frame:
    # polygon
    fill_frame(frame)

    for contour in features.contours:
        if cv2.contourArea(contour) > 4000:
            perimeter = cv2.arcLength(contour, True)
            e = 0.005 * perimeter
            contour = cv2.approxPolyDP(contour, epsilon=e, closed=True)

            cv2.drawContours(frame, [contour], contourIdx=-1, color=(0, 255, 0), thickness=2)
            cv2.fillPoly(frame, [contour], (0, 0, 0))

    return frame


def main():
    while True:
        ret, frame = cap.read()

        features = detect_features(frame)
        frame = render_image(frame, features)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
