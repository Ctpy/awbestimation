import scapy_util
import awb_estimation
import sys
import matplotlib.pyplot as mp


def send_receive_train(ip, packet_train, transmission_interval, verbose=True):
    try:
        packet_train_response, unanswered = scapy_util.send_receive_train(ip, awb_estimation.generate_packet_train(1, packet_train), transmission_interval, verbose)
        round_trip_times = scapy_util.calculate_round_trip_time(packet_train_response)
        mp.plot([range(len(packet_train_response))], [round_trip_times])
        mp.ylabel("Round trip time in second")
        mp.xlabel("Packet index")
        mp.show()
        mp.savefig('rtt.png')
    except Exception as e:
        print(e)
        sys.exit()


if __name__ == '__main__':
    send_receive_train(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
