from pythonosc.udp_client import SimpleUDPClient
from utility import Features


class OSC_Sender:
    def __init__(self, ip: str, port: int):
        self.client = SimpleUDPClient(address=ip, port=port)

    def send_features(self, features: Features):
        if len(features.contours) == 0:
            return

        scaling_width = 16384 / 1920
        scaling_height = 16384 / 1200

        self.client.send_message('/user/1/value', features.center_points[0][0] * scaling_width)
        self.client.send_message('/user/2/value', features.center_points[0][1] * scaling_height)
        self.client.send_message('/user/3/value', features.black_white_ratio * 16384)

