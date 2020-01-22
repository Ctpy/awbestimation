import time
import scapy_util
import trend
import sys
import math
import pcap_util
import utility
import matplotlib.pyplot as mp
from subprocess import PIPE
import subprocess
import globals
import numpy as np

mp.switch_backend('agg')


def estimate_available_bandwidth(target, capacity, resolution, verbose=False):
    """
    Estimate the available bandwidth to the given target.
    Higher resolution will impact the performance.

    :param target -- IP of target to estimate
    :param capacity -- Bottleneck/Link capacity to the target
    :param resolution -- Accuracy of the estimation
    :param verbose -- more output
    """
    capacity *= 1000000
    utility.print_verbose("Capacity is :" + str(capacity) + "bit", verbose)
    utility.print_verbose("Start available bandwidth estimation", verbose)
    # Config Data here
    utility.print_verbose("Initializing parameters", verbose)
    current_awb = capacity   # start at 75% of capacity
    awb_min = 0  # Check if smaller 0
    awb_max = capacity  # Check if greater 100
    pct_trend_list = []
    pdt_trend_list = []
    trend_list = []
    # In Mbits
    percentage = 0.5
    transmission_rate = capacity * percentage
    print("Transmission_rate: " + str(transmission_rate))
    # In Byte
    packet_size = 1500 * 8
    # Numbers of packets per train
    train_length = calculate_train_length(transmission_rate, packet_size)
    utility.print_verbose("Train_length :" + str(train_length), verbose)
    current_ack_number = 1
    transmission_interval = calculate_transmission_interval(train_length)
    # Probe starts here
    utility.print_verbose("tcpdump", verbose)
    no_trend_counter = 0
    step = transmission_rate / 2.0
    m = 0
    c = 0
    for i in range(12):
        print("------------Iteration {}-----------".format(i))
        if i > 0:
            if m > 0.00001:
                transmission_rate = transmission_rate - step
                step /= 2.0
                awb_max = transmission_rate
                no_trend_counter = 0
            elif m < 0.00001 and no_trend_counter < 2:
                no_trend_counter += 1
                transmission_rate = transmission_rate + step
                step /= 2.0
                awb_min = transmission_rate
            else:
                break
            train_length = calculate_train_length(transmission_rate, packet_size)
            transmission_interval = calculate_transmission_interval(train_length)
        print("Currently running with these Parameters: ")
        utility.print_verbose("Transmission_interval: " + str(transmission_interval) + ":s", verbose)
        utility.print_verbose("Generating packet_train", verbose)
        # if transmission_interval < globals.MIN_TRANSMISSION_INTERVAL:
        #    transmission_interval = globals.MIN_TRANSMISSION_INTERVAL
        # TODO recalc
        packet_train_numbers = generate_packet_train(current_ack_number, train_length)
        tcpdump_filter = generate_tcpdump_filter(packet_train_numbers)
        template = '-i leftHost-eth0 -tt -U  -w sender2.pcap'
        template = template.split(' ')
        f = ['tcpdump']
        f.extend(template)
        f.extend([tcpdump_filter])
        p = subprocess.Popen(f, stdout=subprocess.PIPE)
        time.sleep(1)

        last_ack_number = packet_train_numbers[-1] + 40
        utility.print_verbose("Start transmission", verbose)
        packet_train_response, unanswered = scapy_util.send_receive_train(target, packet_train_numbers, transmission_interval, verbose)
        utility.print_verbose("Transmission finished", verbose)
        # sort train by seq number
        utility.print_verbose("Calculating RTT", verbose)
        packet_train_response.sort(key=lambda packet: packet[1].seq)
        round_trip_times = scapy_util.calculate_round_trip_time(packet_train_response)
        time.sleep(1)
        pcap_util.convert_to_csv('sender2.pcap', 'sender2.csv', packet_train_numbers)
        timestamps_tcpdump, unanswered_list_tcpdump, packet_l = pcap_util.analyze_csv('sender2.csv', packet_train_numbers)
        print(timestamps_tcpdump)
        # Plot round trip times
        # plot_results(packet_train_response, round_trip_times, 'rtt{}.png'.format(i), True)
        # mp.plot(*zip(*np.array([(sent_time, rtt_tcpdump) for sent_time,r,rtt_tcpdump in timestamps_tcpdump].sort(key=lambda x:x[0]))))
        round_trip_times.sort(key=lambda x:x[0])
        mp.plot(*zip(*timestamps_tcpdump), linestyle= '--', color='red', label="tcpdump")
        mp.plot(*zip(*round_trip_times), linestyle=':', color='blue', label="scapy")
        mp.ylabel("Round trip time in second")
        mp.xlabel("Time in seconds")
        mp.legend(loc='upper right')
        # calculate trend
        mp.title("Rate {} bit/s ".format(transmission_rate))
        mp.savefig('rtt{}.svg'.format(i), format='svg')
        mp.show()
        mp.clf()
        mp.plot(*zip(*timestamps_tcpdump), linestyle= '--', color='red', label="tcpdump")
        filtered_timestamps_scapy, filtered = trend.decreasing_trend_filter(round_trip_times, verbose)
        # mp.plot(*zip(*filtered_timestamps_scapy), linestyle='-', color='green', label="filtered scapy")
        filtered_timestamps_tcpdump, filtered = trend.decreasing_trend_filter(timestamps_tcpdump, verbose)
        print(filtered_timestamps_tcpdump)
        mp.plot(*zip(*filtered_timestamps_tcpdump), linestyle='-.', color='purple', label="filtered tcpdump")
        sent_time, rtt = zip(*filtered_timestamps_tcpdump)
        A = np.vstack([np.array(list(sent_time)), np.ones(len(sent_time))]).T
        m, c = np.linalg.lstsq(A, np.array(list(rtt)), rcond=None)[0]
        mp.plot(np.array(list(sent_time)),np.array(list(sent_time)) * m + c, 'blue', label="OLS")
        utility.print_verbose("Filtered out: {}".format(len(filtered)), verbose)
        utility.print_verbose(filtered, verbose)
        print("Slope: " + str(m))
        current_ack_number = last_ack_number
        c, m = trend.robust_regression_filter(filtered_timestamps_tcpdump, m, c)
        mp.plot(np.array(list(sent_time)), np.array(list(sent_time)) * m + c, 'green', label="IRLS")
        mp.legend(loc='upper right')
        mp.tick_params(axis='x', which='major')
        mp.title("{} bit/s".format(transmission_rate))
        mp.savefig('rtt_filtered{}.svg'.format(i), format='svg')
        mp.show()
        mp.clf()
        # # wait that fleets dont interfere
        time.sleep(1)
    # Terminate and return
    print ("[" + str(awb_min) + "," + str(awb_max) + "]")
    return awb_min, awb_max


def generate_tcpdump_filter(packet_train):
    template = "(tcp[4:4] = {} and tcp[8:4] = 0) or (tcp[4:4] = 0 and tcp[8:4] = {})"
    tcpdump_filter = ""
    for i, packet in enumerate(packet_train):
        if i > 0:
            tcpdump_filter += " or "
        tcpdump_filter += template.format(packet, packet)
    return tcpdump_filter


def generate_packet_train(starting_number, size):
    """
    Generate Acknowledgement numbers for a packet train.
    """
    print(size)
    print(type(size))
    train = []
    for i in range(size):
        train.append(1 + starting_number + 40 * i)
    return train


def calculate_parameters(trend, train_length, transmission_interval, min_awb, max_awb, packet_loss_rate, packet_size):
    """
    Adjust awb_estimate parameters according on the result.

    :param trend -- the trend of current iteration
    :param train_length -- packet train length
    :param transmission_interval -- send time between two consecutive packets
    :param min_awb - minimal awb bound
    :param max_awb - maximal awb bound
    :param packet_loss_rate - packet loss rate
    :param packet_size -- size of each packet in Byte
    """

    # TODO

    return


def calculate_train_length(transmission_rate, packet_size):
    return int(math.ceil(transmission_rate/packet_size))


def calculate_transmission_interval(train_length):
    return 1.0/float(train_length)


def plot_results(packet_train_response, round_trip_times, filename='rtt.png', clear=False):
    mp.plot(np.array(range(len(packet_train_response))), np.array(round_trip_times), 'o')
    mp.ylabel("Round trip time in second")
    mp.xlabel("Time in seconds")
    mp.savefig(filename, format='png')
    mp.show()
    if clear:
        mp.clf()


if __name__ == '__main__':
    estimate_available_bandwidth(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), True)
    # estimate_available_bandwidth('google.com', 5, 10, True)
