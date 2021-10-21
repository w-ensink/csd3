# partly inspired by: https://automaticaddison.com/how-to-draw-contours-around-objects-using-opencv/
#                and: https://www.programcreek.com/python/example/89328/cv2.approxPolyDP

import numpy as np
import cv2
from object_detection_kmeans import process_contours, process_frame
from perspective_transformer import get_frame_dimensions
from utility import Frame, Features

cap = cv2.VideoCapture(1)
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


def get_black_white_ratio(frame):
    bw_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    frame_dimensions = get_frame_dimensions(bw_frame)
    non_zeros = cv2.findNonZero(bw_frame)
    if non_zeros is None:
        return 0.0
    return len(non_zeros) / ((frame_dimensions[0] * frame_dimensions[1]) + 0.0001)


# TODO: get center point detection working
def detect_features(frame: Frame) -> Features:
    contours = get_contours(frame)
    fill_frame(frame)
    for contour in contours:
        if cv2.contourArea(contour) > 4000:
            perimeter = cv2.arcLength(contour, True)
            e = 0.001 * perimeter
            contour = cv2.approxPolyDP(contour, epsilon=e, closed=True)

            cv2.drawContours(frame, [contour], contourIdx=-1, color=(0, 255, 0), thickness=6)
            cv2.fillPoly(frame, [contour], (0, 0, 0))

    black_white_ratio = get_black_white_ratio(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    center_point = get_center_point_bw(frame)
    return Features([center_point], contours, black_white_ratio)


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
    fill_frame(frame)

    for contour in features.contours:
        if cv2.contourArea(contour) > 4000:
            perimeter = cv2.arcLength(contour, True)
            e = 0.001 * perimeter
            contour = cv2.approxPolyDP(contour, epsilon=e, closed=True)

            cv2.drawContours(frame, [contour], contourIdx=-1, color=(0, 255, 0), thickness=6)
            cv2.fillPoly(frame, [contour], (0, 0, 0))

    cp = features.center_points[0]
    cv2.circle(frame, (int(cp[0]), int(cp[1])), 5, (0, 0, 255), 5)

    return frame


def get_center_point_bw(frame):
    cp = process_frame(frame)[0]
    dims = get_frame_dimensions(frame)
    return (cp[0] / dims[0]) * 16384, (cp[1] / dims[1]) * 16384


# get contours -> draw bw contours -> bw detect & centre point detect
def main():
    video = cv2.VideoCapture('../assets/screen_recording.mp4')

    while True:
        ret, frame = video.read()

        features = detect_features(frame)
        print(f'bw ratio: {features.black_white_ratio}, center: {features.center_points[0]}')
        frame = render_image(frame, features)

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
