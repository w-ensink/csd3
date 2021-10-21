import sys

from perspective_transformer import transform_frame, user_pressed_esc, get_source_transform_points
import cv2
from feature_detection import detect_features, render_image
from utility import TransformInfo, parse_config
from osc_sender import OSC_Sender
from osc_recorder import OSC_Recorder


def main():
    config = parse_config(sys.argv[1])
    camera = cv2.VideoCapture(config['camera'])
    transform_info = TransformInfo(display_points=get_source_transform_points(config['transform_data']),
                                   target_width=config['width'],
                                   target_height=config['height'])

    osc_sender = OSC_Sender(config['osc_ip_address'], config['osc_port'])

    record = config['enable_recording']
    assert type(record) == bool

    if record:
        recorder = OSC_Recorder(config['recording_file'])

    while True:
        _, frame = camera.read()
        frame = transform_frame(frame, transform_info)
        features = detect_features(frame)
        osc_sender.send_features(features)

        if record:
            recorder.send_features(features)

        frame = render_image(frame, features)

        cv2.imshow('image', frame)

        if user_pressed_esc():
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
