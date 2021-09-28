from utility import Features
import time


class OSC_Recorder:
    def __init__(self, file: str):
        self.file = file
        self.data = []
        self.start_time = time.time()

    def send_features(self, features: Features):
        if len(features.contours) == 0:
            return
        contour = features.contours[0][0][0]
        print(f'x: {contour[0]}, y: {contour[1]}')

        scaling_width = 16384 / 1920
        scaling_height = 16384 / 1200

        self.send_message('/user/1/value', float(contour[0]) * scaling_width)
        self.send_message('/user/2/value', float(contour[1]) * scaling_height)

    def send_message(self, address, value):
        time_point = time.time() - self.start_time
        self.data.append(f'{time_point}|{address}|{value}')

    def __del__(self):
        with open(self.file, 'w') as f:
            for m in self.data:
                f.write(f'{m}\n')
