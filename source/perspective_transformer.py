import cv2
import numpy as np
import unittest


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


def get_transform_matrix():
    with open('transform-matrix.txt', 'r') as f:
        points = []
        for _ in range(4):
            line = f.readline()
            x, y = line.split(',')
            points.append([int(x), int(y)])
        points = sort_points(points)
        return np.float32(points)


def main():
    cap = cv2.VideoCapture(0)
    transform_matrix = get_transform_matrix()

    while cap.isOpened():
        ret, frame = cap.read()
        width = 1920 #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = 1200 #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dst = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        matrix = cv2.getPerspectiveTransform(transform_matrix, dst)
        frame = cv2.warpPerspective(frame, matrix, (1920, 1200))

        cv2.imshow('frame', frame)

        if user_pressed_esc():
            break

    cap.release()
    cv2.destroyAllWindows()


# links boven, rechts boven, links onder, rechts onder

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
