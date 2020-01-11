import time
import scapy_util
import pcap_util
import trend
import subprocess as sp
import sys
import utility
import matplotlib.pyplot as mp
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
    capacity *= 1000000 * 8
    utility.print_verbose("Start available bandwidth estimation", verbose)
    # Config Data here
    utility.print_verbose("Initializing parameters", verbose)
    current_awb = capacity * 0.75  # start at 75% of capacity
    awb_min = (1 - resolution) * current_awb  # Check if smaller 0
    awb_max = (1 - resolution) * current_awb  # Check if greater 100
    pct_trend_list = []
    pdt_trend_list = []
    trend_list = []
    # In Mbits
    transmission_rate = capacity * 0.75
    # In Byte
    packet_size = 1500
    # Numbers of packets per train
    train_length = 100
    current_ack_number = 1
    transmission_interval = calculate_transmission_interval(transmission_rate, train_length, packet_size)
    # Probe starts here
    for i in range(12):
        print("Currently running with these Parameters: ")
        utility.print_verbose("Transmission_interval: " + str(transmission_interval) + ":s", verbose)
        utility.print_verbose("Generating packet_train", verbose)
        # if transmission_interval < globals.MIN_TRANSMISSION_INTERVAL:
        #    transmission_interval = globals.MIN_TRANSMISSION_INTERVAL
        # TODO recalc
        packet_train_numbers = generate_packet_train(current_ack_number, train_length)
        last_ack_number = packet_train_numbers[-1] + 40
        utility.print_verbose("Start transmission", verbose)
        packet_train_response, unanswered_list = scapy_util.send_receive_train(target, packet_train_numbers,
                                                                               transmission_interval, verbose)
        utility.print_verbose("Transmission finished", verbose)
        # sort train by seq number
        utility.print_verbose("Calculating RTT", verbose)
        packet_train_response.sort(key=lambda packet: packet[1].seq)
        round_trip_times = scapy_util.calculate_round_trip_time(packet_train_response)
        utility.print_verbose("Packet_loss_rate: " + str(len(unanswered_list) / train_length), verbose)

        # Plot round trip times
        plot_results(packet_train_response, round_trip_times, 'rtt{}.png'.format(i), True)
        transmission_interval /= 2

        # calculate trend

        # trend_state, pct_trend, pdt_trend = trend.calculate_trend(timestamps, packet_loss, train_length)
        # trend_list.append(trend_state)
        # pct_trend_list.append(pct_trend)
        # pdt_trend_list.append(pdt_trend)
        #
        # # fabprobe logic
        # if i > 1:
        #     if (trend_list[-1] == "NOCHANGE" or trend_list[-1] == "UNCL") and trend_list[-2] == "INCR":
        #         return awb_min, awb_max
        #
        # # Adjust state variables
        # current_awb, train_length, transmission_interval = calculate_parameters(trend_state, train_length, transmission_interval, awb_min, awb_max, packet_loss_rate, packet_size)
        # if current_awb > awb_max:
        #     awb_max = current_awb
        # elif current_awb < awb_min:
        #     awb_min = current_awb
        current_ack_number = last_ack_number
        # # wait that fleets dont interfere
        # time.sleep(1)
    # Terminate and return

    return awb_min, awb_max


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
    new_awb = 0.0
    if trend == "INCR":
        new_awb = min_awb
        # Ask: increase transmission_interval or decrease train length?
        if packet_loss_rate > 0.5:
            train_length = calculate_train_length(new_awb, transmission_interval, packet_size)
        else:
            transmission_interval = calculate_transmission_interval(new_awb, train_length, packet_size)
    elif trend == "UNLC" or trend == "NOCHANGE":
        # increase awb
        if packet_loss_rate > 0.5:
            train_length = calculate_train_length(new_awb, transmission_interval, packet_size)
        else:
            transmission_interval = calculate_transmission_interval(new_awb, train_length, packet_size)
    return new_awb, train_length, transmission_interval


def calculate_train_length(transmission_rate, transmission_interval, packet_size):
    return (transmission_interval * transmission_rate) / packet_size


def calculate_transmission_interval(transmission_rate, train_length, packet_size):
    return (train_length * packet_size) / transmission_rate


def plot_results(packet_train_response, round_trip_times, filename='rtt.png', clear=False):
    mp.plot(np.array(range(len(packet_train_response))), np.array(round_trip_times))
    mp.ylabel("Round trip time in second")
    mp.xlabel("Packet index")
    mp.savefig(filename, format='png')
    mp.show()
    if clear:
        mp.clf()

if __name__ == '__main__':
    estimate_available_bandwidth(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), True)
    # estimate_available_bandwidth('google.com', 5, 10, True)
