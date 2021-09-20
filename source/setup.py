import sys
import cv2
import yaml

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


def write_transform_to_file(file_path: str):
    with open(file_path, 'w') as f:
        for x, y in draw_positions:
            f.writelines(f'{x},{y}\n')


def parse_config(file_path: str):
    with open(file_path, 'r') as file:
        return yaml.full_load(file)


def main():
    config = parse_config(sys.argv[1])
    cap = cv2.VideoCapture(config['camera'])

    while cap.isOpened():
        ret, frame = cap.read()
        frame = draw_lines(frame)

        cv2.imshow('frame', frame)
        cv2.setMouseCallback('frame', mouse_callback)

        if user_pressed_esc():
            break

    cap.release()
    cv2.destroyAllWindows()
    write_transform_to_file(config['transform_data'])


if __name__ == '__main__':
    main()
