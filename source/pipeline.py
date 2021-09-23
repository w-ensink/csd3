import sys

from perspective_transformer import parse_config, transform_frame, user_pressed_esc, get_source_transform_points
import cv2
from feature_detection import process_image


def main():
    config = parse_config(sys.argv[1])
    camera = cv2.VideoCapture(config['camera'])
    width = config['width']
    height = config['height']
    transform_points = get_source_transform_points(config['transform_data'])

    while True:
        _, frame = camera.read()
        frame = transform_frame(frame,
                                transform_points=transform_points,
                                target_width=width,
                                target_height=height)
        frame = process_image(frame)

        cv2.imshow('image', frame)

        if user_pressed_esc():
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()