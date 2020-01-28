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


def estimate_available_bandwidth(target, rate=1.0, resolution=10, verbose=False):
    """
    Estimate the available bandwidth to the given target.
    Higher resolution will impact the performance.

    :param target -- IP of target to estimate
    :param rate -- Bottleneck/Link capacity to the target
    :param resolution -- Accuracy of the estimation
    :param verbose -- more output
    """
    rate *= 1000000
    utility.print_verbose("Capacity is :" + str(rate) + "bit", verbose)
    utility.print_verbose("Start available bandwidth estimation", verbose)
    # Config Data here
    utility.print_verbose("Initializing parameters", verbose)
    current_awb = rate   # start at 75% of capacity
    awb_min = 0  # Check if smaller 0
    awb_max = rate  # Check if greater 100
    pct_trend_list = []
    pdt_trend_list = []
    trend_list = []
    # In Mbits
    percentage = 0.5
    transmission_rate = rate * percentage
    print("Transmission_rate: " + str(transmission_rate))
    # In Byte
    packet_size = 1500 * 8
    # Numbers of packets per train
    train_length = 100
    current_ack_number = 1
    transmission_interval = calculate_transmission_interval(rate, packet_size)
    # Probe starts here
    utility.print_verbose("tcpdump", verbose)
    m = 0
    c = 0
    # send N=12 streams
    pdt = []
    pct = []
    for i in range(12):
        print("------------Iteration {}-----------".format(i))
        utility.print_verbose("Current Parameters \n Period: {}\n Train length: {}\n Packet size: {}".format(transmission_interval, train_length, packet_size), verbose)
        utility.print_verbose("Generating packet_train", verbose)
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
        packet_train_response, unanswered = scapy_util.send_receive_train(target, packet_train_numbers, transmission_interval, 3,verbose)
        utility.print_verbose("Transmission finished", verbose)
        # sort train by seq number
        # TODO: divide into substreams
        utility.print_verbose("Calculating RTT", verbose)
        packet_train_response.sort(key=lambda packet: packet[1].seq)
        round_trip_times = scapy_util.calculate_round_trip_time(packet_train_response)
        time.sleep(1)
        pcap_util.convert_to_csv('sender2.pcap', 'sender2.csv', packet_train_numbers)
        timestamps_tcpdump, mean, packet_loss = pcap_util.analyze_csv('sender2.csv', packet_train_numbers)
        pct.append(trend.pct_metric(zip(*timestamps_tcpdump)[1]))
        pdt.append(trend.pdt_metric(zip(*timestamps_tcpdump)[1]))
        utility.print_verbose("PDT: {}".format(pdt), verbose)
        utility.print_verbose("PCT: {}".format(pct), verbose)
        # Plot round trip times
        mp.figure(figsize=(20,8))
        round_trip_times.sort(key=lambda x:x[0])
        mp.plot(*zip(*timestamps_tcpdump), linestyle= '--', color='red', label="tcpdump")
        mp.plot(*zip(*round_trip_times), linestyle=':', color='blue', label="scapy")
        mp.ylabel("Round trip time in second")
        mp.xlabel("Time in seconds")
        mp.legend(loc='upper right')
        mp.ylim(0, 2)
        # calculate trend--------
        mp.title("Rate {} bit/s ".format(transmission_rate))
        mp.savefig('rtt{}.svg'.format(i), format='svg')
        mp.show()
        mp.clf()
        mp.figure(figsize=(20,8))
        mp.plot(*zip(*timestamps_tcpdump), linestyle= '--', color='red', label="tcpdump", marker='s')
        filtered_timestamps_scapy, filtered, s = trend.decreasing_trend_filter(round_trip_times, verbose)
        # mp.plot(*zip(*filtered_timestamps_scapy), linestyle='-', color='green', label="filtered scapy")
        filtered_timestamps_tcpdump, filtered, s = trend.decreasing_trend_filter(timestamps_tcpdump, verbose)
        mp.plot(*zip(*filtered_timestamps_tcpdump), linestyle='-.', color='purple', label="filtered tcpdump", marker="x")
        sent_time, rtt = zip(*filtered_timestamps_tcpdump)
        A = np.vstack([np.array(list(sent_time)), np.ones(len(sent_time))]).T
        m, c = np.linalg.lstsq(A, np.array(list(rtt)), rcond=None)[0]
        mp.plot(np.array(list(sent_time)),np.array(list(sent_time)) * m + c, 'blue', label="OLS")
        utility.print_verbose("Filtered out: {}".format(len(filtered)), verbose)
        print("Slope: " + str(m))
        current_ack_number = last_ack_number
        if len(filtered) < 50:
            c, m = trend.robust_regression_filter(filtered_timestamps_tcpdump, m, c)
            mp.plot(np.array(list(sent_time)), np.array(list(sent_time)) * m + c, 'green', label="IRLS", marker='o')
        mp.legend(loc='upper right')
        mp.tick_params(axis='x', which='major')
        mp.title("{} bit/s".format(transmission_rate))
        mp.savefig('rtt_filtered{}.svg'.format(i), format='svg')
        mp.show()
        mp.clf()
        # # wait that fleets dont interfere
        time.sleep(mean)
    # plot PCT PDT
    mp.ylim(-1,1)
    mp.plot(np.arange(1,13), pdt, linestyle='-', marker='o')
    mp.axhline(y=0.55, linestyle='--')
    mp.axhline(y=0.45, linestyle='--')
    mp.xlabel("# Packet Train")
    mp.ylabel("PDT metric")
    mp.title("PDT metric")
    mp.savefig('pdt_metric.svg', format='svg')
    mp.clf()
    mp.ylim(0,1)
    mp.plot(np.arange(1,13), pct, linestyle='-', marker='o')
    mp.axhline(y=0.66, linestyle='--')
    mp.axhline(y=0.54, linestyle='--')
    mp.xlabel("# Packet Train")
    mp.ylabel("PCT metric")
    mp.title("PCT metric")
    mp.savefig('pct_metric.svg', format='svg')
    mp.clf()
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

    # TODO: Implement rate adjustment algorithm

    return


def calculate_transmission_interval(rate, packet_size):
    # TODO Base calc on current target rate
    return packet_size/rate


def plot_results(packet_train_response, round_trip_times, filename='rtt.png', clear=False):
    mp.plot(np.array(range(len(packet_train_response))), np.array(round_trip_times), 'o')
    mp.ylabel("Round trip time in second")
    mp.xlabel("Time in seconds")
    mp.savefig(filename, format='png')
    mp.show()
    if clear:
        mp.clf()


def calculate_median(timestamps):
    sum = 0.0
    for i in timestamps:
        sum += i
    return sum/len(timestamps)


if __name__ == '__main__':
    estimate_available_bandwidth(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), True)
    # estimate_available_bandwidth('google.com', 5, 10, True)
