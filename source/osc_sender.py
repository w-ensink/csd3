from pythonosc.udp_client import SimpleUDPClient
from utility import Features


class OSC_Sender:
    def __init__(self, ip: str, port: int):
        self.client = SimpleUDPClient(address=ip, port=port)

    def send_features(self, features: Features):
        if len(features.contours) == 0:
            return

        scaling_width = 16384 / 1000
        scaling_height = 16384 / 700
        x = features.center_points[0][0] * scaling_width
        y = features.center_points[0][1] * scaling_height
        print(f'center point: {x}, {y}')
        self.client.send_message('/user/1/value', int(x))
        self.client.send_message('/user/2/value', int(y))

