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
    tcp_ack = IP(dst=ip, id=ack_number) / TCP(ack=ack_number, flags='A') / Raw(load=payload)
    return tcp_ack


def generate_packet_train(ip, ack_numbers, payload):
    print("In generate_packet_train")
    print(ack_numbers)
    train = []
    for ack_number in ack_numbers:
        train.append(generate_packet(ip, ack_number, payload))
    return train


def send_train(ip, packet_train, transmission_interval, verbose):
    send(generate_packet_train(ip, packet_train, 'x' * 1452), inter=transmission_interval, verbose=verbose)


def send_receive_train(ip, packet_train_size, transmission_interval, timeout=1, verbose=False):
    ack_numbers = awb_estimation.generate_packet_train(1, packet_train_size)
    packet_train = generate_packet_train(ip, ack_numbers, '')
    ans, unans = sr(packet_train, inter=transmission_interval, timeout=timeout)
    print(ans.show())
    for pkt in ans:
        print(pkt[TCP].ack)
    print("packet_loss_rate: " + str(len(unans)/packet_train_size))


if __name__ == '__main__':
    # send_receive_train('google.com', 100, 0.1, True)
    send_receive_train(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
