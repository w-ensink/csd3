from pythonosc.udp_client import SimpleUDPClient
from utility import Features


class OSC_Sender:
    def __init__(self, ip: str, port: int):
        self.client = SimpleUDPClient(address=ip, port=port)

    def send_features(self, features: Features):
        if len(features.contours) == 0:
            return
        contour = features.contours[0][0][0]
        print(f'x: {contour[0]}, y: {contour[1]}')

        scaling_width = 16384 / 1920
        scaling_height = 16384 / 1200

        self.client.send_message('/user/1/value', float(contour[0]) * scaling_width)
        self.client.send_message('/user/2/value', float(contour[1]) * scaling_height)

