from pythonosc.udp_client import SimpleUDPClient
from utility import Features

client: SimpleUDPClient = None


def setup(ip_address: str, port: int):
    global client
    client = SimpleUDPClient(address=ip_address, port=port)


def send_features(features: Features):
    assert client is not None
    if len(features.contours) == 0:
        return
    contour = features.contours[0][0][0]
    print(f'x: {contour[0]}, y: {contour[1]}')

    scaling_width = 16384 / 1920
    scaling_height = 16384 / 1200

    # client.send_message('/user/1/value', float(contour[0]) * scaling_width)
    # client.send_message('/user/2/value', float(contour[1]) * scaling_height)
    client.send_message('/user/1/value', float(features.center_points[0] * scaling_width))
    client.send_message('/user/1/value', float(features.center_points[1] * scaling_height))
