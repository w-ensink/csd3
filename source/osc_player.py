import sys
import time

from pythonosc.udp_client import SimpleUDPClient
from utility import parse_config


def parse_line(line: str):
    line = line.split('|')
    if len(line) != 3:
        return None

    return line[0], line[1], line[2]


def parse_recording_file(file: str):
    with open(file, 'r') as f:
        return [parse_line(l) for l in f.read().split('\n') if l != '']


def main():
    config = parse_config(sys.argv[1])
    sender = SimpleUDPClient(config['osc_ip_address'], config['osc_port'])
    file = config['recording_file']
    events = parse_recording_file(file)

    start_time = time.time()
    current_time = 0

    for time_stamp, address, message in events:
        while current_time < float(time_stamp):
            time.sleep(0.01)
            current_time = time.time() - start_time
        sender.send_message(address, message)
        print(f'sendeing {address}: {message}')


if __name__ == '__main__':
    main()
