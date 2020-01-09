from scapy.layers.inet import *
from scapy.packet import Raw
from scapy.sendrecv import *
import utility
import threading

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
    for i in len(ack_numbers):
        train.append(generate_packet(ip, ack_numbers[i], payload))
    return train


def send_train(ip, packet_train, transmission_interval, verbose):
    send(generate_packet_train(ip, packet_train, 'x' * 1452), inter=transmission_interval, verbose=verbose)


def send_receive_train(ip, packet_train, transmission_interval, verbose):
    event = threading.Event()
    filter = "tcp"
    sniff_queue = queue.Queue()
    sniff_thread = threading.Thread(target=sniff_on_event, args=(event, filter, sniff_queue, verbose,))
    sniff_thread.start()
    sr(generate_packet_train(ip, packet_train, 'x' * 1452), inter=transmission_interval, verbose=verbose)
    event.set()
    packets_list = list(sniff_queue)
    print(packets_list)


def sniff_on_event(event, filter, sniff_queue, verbose):
    packets = sniff(filter=filter, stop_filter=lambda p: event.is_set())
    sniff_queue.put(packets)
    utility.print_verbose("Stop sniffing", verbose)
    return packets


if __name__ == '__main__':
    generate_packet("google.com", 1000, 'x' * 1500)
