from pythonosc.udp_client import SimpleUDPClient
from utility import Features

client: SimpleUDPClient = None


def setup(ip_address: str, port: int):
    global client
    client = SimpleUDPClient(address=ip_address, port=port)


def send_features(features: Features):
    assert client is not None
    client.send_message('/center/x', features.center_points[0][0])
    client.send_message('/center/y', features.center_points[0][1])