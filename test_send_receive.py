import scapy_util
import awb_estimation
import sys


def send_receive_train(ip, packet_train, transmission_interval, verbose=True):

    scapy_util.send_receive_train(ip, awb_estimation.generate_packet_train(1, packet_train), transmission_interval, verbose)


if __name__ == '__main__':
    send_receive_train('google.com', 10, 0.1)
    # send_receive_train(sys.argv[1], sys.argv[2], sys.argv[3], True)
