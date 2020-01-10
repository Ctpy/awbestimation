import scapy_util
import awb_estimation
import sys
import matplotlib
import matplotlib.pyplot as mp
mp.switch_backend('agg')
import numpy as np


def send_receive_train(ip, packet_train_length, transmission_interval, verbose=True):
    try:
        packet_train_response, unanswered = scapy_util.send_receive_train(ip, packet_train_length, transmission_interval, verbose)
        for pkt in packet_train_response:
            print(pkt[1].seq)
        packet_train_response.sort(key=lambda pkt: pkt[1].seq)
        for pkt in packet_train_response:
            print(pkt[1].seq)
        round_trip_times = scapy_util.calculate_round_trip_time(packet_train_response)
        print(len(round_trip_times))
        print(len(packet_train_response))

        mp.plot(np.array(range(len(packet_train_response))), np.array(round_trip_times))
        mp.ylabel("Round trip time in second")
        mp.xlabel("Packet index")
        mp.savefig('rtt.png', format='png')
        mp.show()
    except Exception as e:
        print(e)
        sys.exit()


if __name__ == '__main__':
    send_receive_train(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
