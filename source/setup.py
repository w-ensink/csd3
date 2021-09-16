import cv2
import numpy as np

cap = cv2.VideoCapture(0)
draw_positions = []
COLOR = (255, 0, 0)


def user_pressed_esc() -> bool:
    return cv2.waitKey(24) == 27


def mouse_callback(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        draw_positions.append((x, y))
        print(f'x: {x}, y: {y}')


def draw_lines(frame):
    if len(draw_positions) == 1:
        frame = cv2.circle(frame, draw_positions[0], 1, COLOR, 2)

    for i in range(len(draw_positions)):
        if i + 1 < len(draw_positions):
            frame = cv2.line(frame, draw_positions[i], draw_positions[i + 1], COLOR, 2)

    return frame


def write_transform_to_file():
    with open('transform-matrix.txt', 'w') as f:
        for x, y in draw_positions:
            f.writelines(f'{x},{y}\n')


def main():
    while cap.isOpened():
        ret, frame = cap.read()
        # Locate points of the documents or object which you want to transform
        pts1 = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
        pts2 = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])

        # Apply Perspective Transform Algorithm
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (500, 600))
        # Wrap the transformed image

        frame = draw_lines(frame)

        cv2.imshow('frame', frame)  # Inital Capture
        cv2.setMouseCallback('frame', mouse_callback)
        # cv2.imshow('frame1', result)  # Transformed Capture

        if user_pressed_esc():
            break

    cap.release()
    cv2.destroyAllWindows()
    write_transform_to_file()


if __name__ == '__main__':
    main()
