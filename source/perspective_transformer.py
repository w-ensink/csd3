# partly inspired by: https://medium.com/analytics-vidhya/opencv-perspective-transformation-9edffefb2143

import cv2
import numpy as np
import unittest
import sys
import yaml

from global_setup import get_frame_dimensions
from utility import Frame, TransformInfo, parse_config


def get_top_left(points: [(int, int)]) -> (int, int):
    points = sorted(points, key=lambda x: x[1])
    return sorted(points[0:2], key=lambda x: x[0])[0]


def get_top_right(points: [(int, int)]) -> (int, int):
    points = sorted(points, key=lambda x: x[1])
    return sorted(points[0:2], key=lambda x: x[0])[1]


def get_bottom_left(points: [(int, int)]) -> (int, int):
    points = sorted(points, key=lambda x: x[1])
    return sorted(points[2:4], key=lambda x: x[0])[0]


def get_bottom_right(points: [(int, int)]) -> (int, int):
    points = sorted(points, key=lambda x: x[1])
    return sorted(points[2:4], key=lambda x: x[0])[1]


def sort_points(points):
    assert len(points) == 4
    top_left = get_top_left(points)
    top_right = get_top_right(points)
    bottom_left = get_bottom_left(points)
    bottom_right = get_bottom_right(points)
    return [top_left, top_right, bottom_left, bottom_right]


def user_pressed_esc() -> bool:
    return cv2.waitKey(24) == 27


def get_source_transform_points(file_path: str):
    with open(file_path, 'r') as f:
        points = []
        for _ in range(4):
            line = f.readline()
            x, y = line.split(',')
            points.append([x, y])
        points = sort_points(points)
        return np.float32(points)


def transform_frame(frame: Frame, transform_info: TransformInfo) -> Frame:
    dst = np.float32([[0, 0],
                      [transform_info.target_width, 0],
                      [0, transform_info.target_height],
                      [transform_info.target_width, transform_info.target_height]])
    dimensions = get_frame_dimensions(frame)
    display_points = np.float32(list([int(x * dimensions[0]), int(y * dimensions[1])] for x, y in transform_info.display_points))
    matrix = cv2.getPerspectiveTransform(display_points, dst)
    return cv2.warpPerspective(frame, matrix, (transform_info.target_width, transform_info.target_height))


def main():
    config = parse_config(sys.argv[1])
    cap = cv2.VideoCapture(config['camera'])
    transform_points = get_source_transform_points(config['transform_data'])
    width = config['width']
    height = config['height']
    transform_info = TransformInfo(transform_points, width, height)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = transform_frame(frame, transform_info)
        cv2.imshow('frame', frame)

        if user_pressed_esc():
            break

    cap.release()
    cv2.destroyAllWindows()


class SortTest(unittest.TestCase):
    def test_sorting(self):
        points = [[0, 0], [400, 0], [0, 640], [400, 640]]
        sorted_points = sort_points([[400, 0], [0, 640], [400, 640], [0, 0]])
        self.assertEqual(sorted_points, points)

    def test_sorting_2(self):
        points = [[384, 118], [1020, 130], [330, 591], [1027, 640]]
        sorted_points = sort_points([[384, 118], [1020, 130], [1027, 640], [330, 591]])
        self.assertEqual(sorted_points, points)


if __name__ == '__main__':
    main()
