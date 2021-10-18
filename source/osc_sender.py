from pythonosc.udp_client import SimpleUDPClient
from utility import Features


class OSC_Sender:
    def __init__(self, ip: str, port: int):
        self.client = SimpleUDPClient(address=ip, port=port)

    def send_features(self, features: Features):
        if len(features.contours) == 0:
            return

        x = features.center_points[0][0]
        y = features.center_points[0][1]
        print(f'center point: {x}, {y}, ratio: {features.black_white_ratio * 16384}')
        self.client.send_message('/user/1/value', int(x))
        self.client.send_message('/user/2/value', int(y))
        self.client.send_message('/user/3/value', int(features.black_white_ratio * 16000))


