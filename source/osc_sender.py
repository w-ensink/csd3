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
    client.send_message('/center/x', float(contour[0]))
    client.send_message('/center/y', float(contour[1]))