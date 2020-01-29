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
import timeit

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
    start = timeit.default_timer()
    rate *= 1000000 *1.5
    utility.print_verbose("Capacity is :" + str(rate) + "bit", verbose)
    utility.print_verbose("Start available bandwidth estimation", verbose)
    # Config Data here
    utility.print_verbose("Initializing parameters", verbose)
    current_awb = rate  # start at 75% of capacity
    awb_min = 0  # Check if smaller 0
    awb_max = rate  # Check if greater 100

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

    # send N=12 streams
    pdt = []
    pct = []
    rtt_list = []
    rtt_train_list = []
    for i in range(12):
        print("------------Iteration {}-----------".format(i))
        utility.print_verbose(
            "Current Parameters \n Period: {}\n Train length: {}\n Packet size: {}".format(transmission_interval,
                                                                                           train_length, packet_size),
            verbose)
        utility.print_verbose("Generating packet_train", verbose)
        packet_train_numbers = generate_packet_train(current_ack_number, train_length)
        last_ack_number = packet_train_numbers[-1] + 40
        utility.print_verbose("Start transmission", verbose)
        packet_train_response, unanswered = scapy_util.send_receive_train(target, packet_train_numbers,
                                                                          transmission_interval, 10, verbose)
        if len(unanswered) == train_length:
            continue
        utility.print_verbose("Transmission finished", verbose)
        # sort train by seq number
        utility.print_verbose("Calculating RTT", verbose)
        packet_train_response.sort(key=lambda packet: packet[1].seq)
        round_trip_times = scapy_util.calculate_round_trip_time(packet_train_response)
        rtt_list.extend(round_trip_times)
        rtt_train_list.append(round_trip_times)
        mean = np.mean(zip(*round_trip_times)[1])
        # calculate pdt and pct metric
        pct.append(trend.pct_metric(zip(*round_trip_times)[1]))
        pdt.append(trend.pdt_metric(zip(*round_trip_times)[1]))
        utility.print_verbose("PDT: {}".format(pdt), verbose)
        utility.print_verbose("PCT: {}".format(pct), verbose)
        # # wait that fleets dont interfere
        time.sleep(mean)

    # plot RTT of all packet trains
    start_sent_time = rtt_list[0][0]
    sent_time, rtt = zip(*rtt_list)
    sent_time = np.array(sent_time)
    mp.plot(sent_time - start_sent_time, rtt, linestyle='-', marker='x')
    mp.xlabel("Sent time in seconds")
    mp.ylabel("Round trip time in seconds")
    mp.savefig('rtt.pdf', format='pdf')
    mp.figure(figsize=(20, 8))
    mp.clf()

    # Determine trend based on PDT/PCT
    increase_pdt = 0
    grey_pdt = 0
    no_trend_pdt = 0
    increase_pct = 0
    grey_pct = 0
    no_trend_pct = 0

    trend_pdt = -1
    trend_pct = -1
    trend_overall = -1
    for i in pdt:
        if i > 0.55:
            increase_pdt += 1
        elif i < 0.45:
            no_trend_pdt += 1
        else:
            grey_pdt += 1

    for i in pct:
        if i > 0.66:
            increase_pct += 1
        elif i < 0.54:
            no_trend_pct += 1
        else:
            grey_pct += 1

    if increase_pdt / len(pdt) > 0.7:
        trend_pdt = 2
    elif no_trend_pdt / len(pdt) > 0.7:
        trend_pdt = 0
    else:
        trend_pdt = 1

    if increase_pct / len(pct) > 0.6:
        trend_pct = 2
    elif no_trend_pct / len(pct) > 0.6:
        trend_pct = 0
    else:
        trend_pct = 1

    if trend_pdt == 2 and trend_pct == 2:
        trend_overall = 2
    elif trend_pdt == 2 and trend_pct == 1:
        trend_overall = 2
    elif trend_pdt == 1 and trend_pct == 2:
        trend_overall = 2
    utility.print_verbose("Trend after PCT/PDT: {}".format(trend_overall),    verbose)
    # Decreasing trend filter
    dt_filtered_pct = []
    dt_filtered_pdt = []
    dt_filtered_train_list = []
    for packet_train in rtt_train_list:
        timestamps, filtered = trend.decreasing_trend_filter(packet_train, False)
        dt_filtered_pct.append(trend.pct_metric(zip(*timestamps)[1]))
        dt_filtered_pdt.append(trend.pdt_metric(zip(*timestamps)[1]))
        utility.print_verbose("Filtered out: {}".format(len(filtered)), verbose)
        dt_filtered_train_list.append(timestamps)

    mp.ylim(-1, 1)
    mp.plot(np.arange(1, len(pdt) + 1), pdt, linestyle='-', marker='o', color='blue', label='Original')
    mp.plot(np.arange(1, len(dt_filtered_pdt) + 1), dt_filtered_pdt, linestyle='-', marker='x', color='red', label='DT Filtered')
    mp.axhline(y=0.55, linestyle='--')
    mp.axhline(y=0.45, linestyle='--')
    mp.xlabel("# Packet Train")
    mp.ylabel("PDT metric")
    mp.title("PDT metric")
    mp.legend(loc='upper right')
    mp.savefig('pdt_metric.svg', format='svg')
    mp.clf()
    mp.ylim(0, 1)
    mp.plot(np.arange(1, len(pct) + 1), pct, linestyle='-', marker='o', color='blue', label='Original')
    mp.plot(np.arange(1, len(dt_filtered_pct) + 1), dt_filtered_pct, linestyle='-', marker='x', color='red', label='DT Filtered')
    mp.axhline(y=0.66, linestyle='--')
    mp.axhline(y=0.54, linestyle='--')
    mp.xlabel("# Packet Train")
    mp.ylabel("PCT metric")
    mp.title("PCT metric")
    mp.legend(loc='upper right')
    mp.savefig('pct_metric.svg', format='svg')

    for i in dt_filtered_pdt:
        if i > 0.55:
            increase_pdt += 1
        elif i < 0.45:
            no_trend_pdt += 1
        else:
            grey_pdt += 1

    for i in dt_filtered_pct:
        if i > 0.66:
            increase_pct += 1
        elif i < 0.54:
            no_trend_pct += 1
        else:
            grey_pct += 1

    if increase_pdt / len(pdt) > 0.6:
        trend_pdt = 2
    elif no_trend_pdt / len(pdt) > 0.6:
        trend_pdt = 0
    else:
        trend_pdt = 1

    if increase_pct / len(pct) > 0.6:
        trend_pct = 2
    elif no_trend_pct / len(pct) > 0.6:
        trend_pct = 0
    else:
        trend_pct = 1

    if trend_pdt == 2 and trend_pct == 2:
        trend_overall = 2
    elif trend_pdt == 2 and trend_pct == 1:
        trend_overall = 2
    elif trend_pdt == 1 and trend_pct == 2:
        trend_overall = 2
    elif trend_pdt == 0 and trend_pct == 0:
        trend_overall = 0
    elif trend_pdt == 0 and trend_pct == 1:
        trend_overall = 0
    elif trend_pdt == 1 and trend_pct == 0:
        trend_overall = 0
    else:
        trend_overall = 1

    utility.print_verbose("Trend after DT filtering: {}".format(trend_overall), verbose)
    # Robust regression filter

    rr_filtered_pct = []
    rr_filtered_pdt = []
    rr_filtered_train_list = []
    for packet_train in dt_filtered_train_list:
        timestamps = trend.robust_regression_filter(zip(*packet_train)[1])
        rr_filtered_pct.append(trend.pct_metric(timestamps))
        rr_filtered_pdt.append(trend.pdt_metric(timestamps))
        rr_filtered_train_list.append(timestamps)

    for i in rr_filtered_pdt:
        if i > 0.55:
            increase_pdt += 1
        elif i < 0.45:
            no_trend_pdt += 1
        else:
            grey_pdt += 1

    for i in rr_filtered_pct:
        if i > 0.66:
            increase_pct += 1
        elif i < 0.54:
            no_trend_pct += 1
        else:
            grey_pct += 1

    if increase_pdt / len(pdt) > 0.6:
        trend_pdt = 2
    elif no_trend_pdt / len(pdt) > 0.6:
        trend_pdt = 0
    else:
        trend_pdt = 1

    if increase_pct / len(pct) > 0.6:
        trend_pct = 2
    elif no_trend_pct / len(pct) > 0.6:
        trend_pct = 0
    else:
        trend_pct = 1

    if trend_pdt == 2 and trend_pct == 2:
        trend_overall = 2
    elif trend_pdt == 2 and trend_pct == 1:
        trend_overall = 2
    elif trend_pdt == 1 and trend_pct == 2:
        trend_overall = 2
    elif trend_pdt == 0 and trend_pct == 0:
        trend_overall = 0
    elif trend_pdt == 0 and trend_pct == 1:
        trend_overall = 0
    elif trend_pdt == 1 and trend_pct == 0:
        trend_overall = 0
    else:
        trend_overall = 1

    utility.print_verbose("Trend after RR filtering: {}".format(trend_overall), verbose)
    # Rate adjustment

    # Terminate and return
    utility.print_verbose("Runtime for 1 fleet: {}s".format(timeit.default_timer() - start), verbose)
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
    return packet_size / rate


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
