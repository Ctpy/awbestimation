from scapy.layers.inet import *
from scapy.packet import Raw
from scapy.sendrecv import *
import awb_estimation
import sys
try:
    import queue
except ImportError:
    import Queue as queue


def generate_packet(ip, ack_number, payload):
    tcp_ack = Ether()/IP(dst=ip, id=ack_number) / TCP(ack=ack_number, flags='A') / Raw(load=payload)
    return tcp_ack


def generate_packet_train(ip, ack_numbers, payload):
    train = []
    for ack_number in ack_numbers:
        train.append(generate_packet(ip, ack_number, payload))
    return train


def send_train(ip, packet_train, transmission_interval, verbose):
    send(generate_packet_train(ip, packet_train, 'x' * 1452), inter=transmission_interval, verbose=verbose)


def send_receive_train(ip, packet_train_numbers, transmission_interval, timeout=10, verbose=False):
    packet_train = generate_packet_train(ip, packet_train_numbers, 'x'*1452)
    ans, unans = srp(packet_train, inter=transmission_interval, timeout=timeout)
    return ans, unans


def calculate_round_trip_time(packet_train):
    round_trip_times = []
    for pkt in packet_train:
        round_trip_times.append((pkt[0].sent_time, pkt[1].time - pkt[0].sent_time))
    return round_trip_times


if __name__ == '__main__':
    # send_receive_train('google.com', 100, 0.1, True)
    calculate_round_trip_time(send_receive_train(sys.argv[1], int(sys.argv[2]), float(sys.argv[3])))
