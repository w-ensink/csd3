import sys
import cv2
import yaml
from contours import get_screen_contour, get_corners


screen_corners = []
saved_screen_corners = []
resolution = (1, 2)
COLOR = (0, 255, 0)


def user_pressed_esc() -> bool:
    return cv2.waitKey(24) == 27


def mouse_callback(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global saved_screen_corners
        saved_screen_corners = screen_corners


def draw_screen_outlines(frame, corners):
    cv2.line(frame, corners[0], corners[1], COLOR, 2)
    cv2.line(frame, corners[1], corners[2], COLOR, 2)
    cv2.line(frame, corners[2], corners[3], COLOR, 2)
    cv2.line(frame, corners[3], corners[0], COLOR, 2)
    return frame


def write_transform_to_file(file_path: str):
    with open(file_path, 'w') as f:
        for x, y in saved_screen_corners:
            x /= resolution[0]
            y /= resolution[1]
            f.writelines(f'{x},{y}\n')


def parse_config(file_path: str):
    with open(file_path, 'r') as file:
        return yaml.full_load(file)


def get_frame_dimensions(frame):
    return len(frame[0]), len(frame)


def main():
    config = parse_config(sys.argv[1])
    cap = cv2.VideoCapture(config['camera'])

    while cap.isOpened():
        ret, frame = cap.read()
        contour = get_screen_contour(frame)
        if len(contour) > 0:
            corners = get_corners(contour)
            if len(corners) == 4:
                global screen_corners, resolution
                screen_corners = corners
                resolution = get_frame_dimensions(frame)
                frame = draw_screen_outlines(frame, corners)

        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', mouse_callback)

        if user_pressed_esc():
            break

    cap.release()
    cv2.destroyAllWindows()
    write_transform_to_file(config['transform_data'])


if __name__ == '__main__':
    main()
