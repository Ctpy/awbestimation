from scapy.layers.inet import *
from scapy.packet import Raw
from scapy.sendrecv import *


def generate_packet(ip, ack_number, payload):
    tcp_ack = IP(dst=ip, id=ack_number) / TCP(ack=ack_number, flags='A') / Raw(load=payload)
    return tcp_ack


def generate_packet_train(ip, ack_numbers: list, payload):
    train = []
    for i in len(ack_numbers):
        train.append(generate_packet(ip, ack_numbers[i], payload))
    return train


def send_train(ip, packet_train, transmission_interval, verbose):
    send(generate_packet_train(ip, packet_train, 'x' * 1452), inter=transmission_interval, verbose=verbose)


if __name__ == '__main__':
    generate_packet("google.com", 1000, 'x' * 1500)
