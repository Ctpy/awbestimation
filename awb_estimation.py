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


def estimate_available_bandwidth(target, rate=1.0, resolution=0.5, verbose=False):
    """
    Estimate the available bandwidth to the given target.
    Higher resolution will impact the performance.

    :param target -- IP of target to estimate
    :param rate -- Bottleneck/Link capacity to the target
    :param resolution -- Accuracy of the estimation
    :param verbose -- more output
    """
    mkr = ['.', ',', 'o', 'x', 'D', 'd', '+', '1', '2', '3', '4', 's', 'h', '*']
    color = ['blue', 'black', 'cyan', 'magenta', 'green', 'yellow', 'red', 'violet', 'brown', 'grey', '#eeefff', 'pink']
    start = timeit.default_timer()
    res = 0.1
    rate *= 10000000
    utility.print_verbose("Capacity is :" + str(rate) + "bit", verbose)
    utility.print_verbose("Start available bandwidth estimation", verbose)
    # Config Data here
    utility.print_verbose("Initializing parameters", verbose)
    current_awb = rate  # start at 75% of capacity
    awb_min = 0  # Check if smaller 0
    awb_max = rate  # Check if greater 100 TODO: increase to max possible rate
    grey_min = 0
    grey_max = 0
    # In Mbits
    transmission_rate = rate * 0.75
    print("Transmission_rate: " + str(transmission_rate))
    # In Byte
    packet_size = 1500 * 8
    # Numbers of packets per train
    train_length = 100
    current_ack_number = 1
    transmission_interval = calculate_transmission_interval(transmission_rate, packet_size)
    # Probe starts here
    iteration_max = 12
    loop_counter = 0
    while loop_counter < iteration_max or abs(awb_min - awb_max) > res:
        # send N=12 streams
        mp.figure(1)
        pdt = []
        pct = []
        rtt_list = []
        rtt_train_list = []
        utility.print_verbose(
            "Current Parameters \n Period: {}\n Train length:     {}\n Packet   size: {}".format(transmission_interval,
                                                                                                 train_length,
                                                                                                 packet_size), verbose)

        for i in range(12):
            print("------------Fleet {} - Iteration {}-----------".format(loop_counter, i))
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
            time.sleep(abs(mean))

        # plot RTT of all packet trains
        start_sent_time = rtt_list[0][0]
        sent_time, rtt = zip(*rtt_list)
        sent_time = np.array(sent_time)
        mp.figure(figsize=(20, 8))
        mp.plot(sent_time - start_sent_time, rtt, color=color[loop_counter], marker=mkr[loop_counter],
                label="{}b/s".format(transmission_rate))
        mp.xlabel("Sent time in seconds")
        mp.ylabel("Round trip time in seconds")
        mp.savefig('plots/rtt{}.pdf'.format(loop_counter), format='pdf')

        # Determine trend based on PDT/PCT
        increase_pdt = 0.0
        grey_pdt = 0.0
        no_trend_pdt = 0.0
        increase_pct = 0.0
        grey_pct = 0.0
        no_trend_pct = 0.0

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
            if i > 0.65:
                increase_pct += 1
            elif i < 0.54:
                no_trend_pct += 1
            else:
                grey_pct += 1
        utility.print_verbose(
            "PCT: INC:{} - NO:{} - GRAY:{} - Len{}".format(increase_pct, no_trend_pct, grey_pct, len(pct)), verbose)
        utility.print_verbose(
            "PDT: INC:{} - NO:{} - GRAY:{} - Len{}".format(increase_pdt, no_trend_pdt, grey_pdt, len(pdt)), verbose)
        if increase_pdt / len(pdt) > 0.65:
            trend_pdt = 2
        elif no_trend_pdt / len(pdt) > 0.65:
            trend_pdt = 0
        else:
            trend_pdt = 1

        if increase_pct / len(pct) > 0.65:
            trend_pct = 2
        elif no_trend_pct / len(pct) > 0.65:
            trend_pct = 0
        else:
            trend_pct = 1
        utility.print_verbose("PCT:{}-PDT:{}".format(trend_pct, trend_pdt), verbose)
        if trend_pdt == 2 and trend_pct == 2:
            trend_overall = 2
        elif trend_pdt == 2 and trend_pct == 1:
            trend_overall = 2
        elif trend_pdt == 1 and trend_pct == 2:
            trend_overall = 2
        utility.print_verbose("Trend after PCT/PDT: {}".format(trend_overall), verbose)
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

        utility.print_verbose("PDT: {}".format(dt_filtered_pdt), verbose)
        utility.print_verbose("PCT: {}".format(dt_filtered_pct), verbose)
        mp.figure(2)
        mp.ylim(-1, 1)
        mp.plot(np.arange(1, len(pdt) + 1), pdt, linestyle='-', marker='o', color='blue', label='Original')
        mp.plot(np.arange(1, len(dt_filtered_pdt) + 1), dt_filtered_pdt, linestyle='-', marker='x', color='red',
                label='DT Filtered')
        mp.axhline(y=0.55, linestyle='--')
        mp.axhline(y=0.45, linestyle='--')
        mp.xlabel("# Packet Train")
        mp.ylabel("PDT metric")
        mp.title("PDT metric")
        mp.legend(loc='upper right')
        mp.savefig('plots/pdt_metric.svg', format='svg')
        mp.clf()
        mp.figure(2)
        mp.ylim(0, 1)
        mp.plot(np.arange(1, len(pct) + 1), pct, linestyle='-', marker='o', color='blue', label='Original')
        mp.plot(np.arange(1, len(dt_filtered_pct) + 1), dt_filtered_pct, linestyle='-', marker='x', color='red',
                label='DT Filtered')
        mp.axhline(y=0.66, linestyle='--')
        mp.axhline(y=0.54, linestyle='--')
        mp.xlabel("# Packet Train")
        mp.ylabel("PCT metric")
        mp.title("PCT metric")
        mp.legend(loc='upper right')
        mp.savefig('plots/pct_metric.svg', format='svg')
        mp.clf()

        increase_pdt = 0.0
        grey_pdt = 0.0
        no_trend_pdt = 0.0
        increase_pct = 0.0
        grey_pct = 0.0
        no_trend_pct = 0.0

        if trend_overall == -1 or trend_overall == 1:
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

            if increase_pdt / len(pdt) > 0.65:
                trend_pdt = 2
            elif no_trend_pdt / len(pdt) > 0.65:
                trend_pdt = 0
            else:
                trend_pdt = 1

            if increase_pct / len(pct) > 0.65:
                trend_pct = 2
            elif no_trend_pct / len(pct) > 0.65:
                trend_pct = 0
            else:
                trend_pct = 1

            if trend_pdt == 2 and trend_pct == 2:
                trend_overall = 2
            elif trend_pdt == 2 and trend_pct == 1:
                trend_overall = 2
            elif trend_pdt == 1 and trend_pct == 2:
                trend_overall = 2

            utility.print_verbose("Trend after DT filtering: {}".format(trend_overall), verbose)
            # Robust regression filter
            if trend_overall == -1 or trend_overall == 1:
                increase_pdt = 0.0
                grey_pdt = 0.0
                no_trend_pdt = 0.0
                increase_pct = 0.0
                grey_pct = 0.0
                no_trend_pct = 0.0
                rr_filtered_pct = []
                rr_filtered_pdt = []
                rr_filtered_train_list = []
                for packet_train in dt_filtered_train_list:
                    timestamps, filtered = trend.robust_regression_filter(zip(*packet_train)[1])
                    rr_filtered_pct.append(trend.pct_metric(timestamps))
                    rr_filtered_pdt.append(trend.pdt_metric(timestamps))
                    rr_filtered_train_list.append(timestamps)
                utility.print_verbose("RR Filtered: {}".format(filtered), verbose)
                utility.print_verbose("PDT: {}".format(rr_filtered_pdt), verbose)
                utility.print_verbose("PCT: {}".format(rr_filtered_pct), verbose)

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

                if increase_pdt / len(pdt) > 0.65:
                    trend_pdt = 2
                elif no_trend_pdt / len(pdt) > 0.65:
                    trend_pdt = 0
                else:
                    trend_pdt = 1

                if increase_pct / len(pct) > 0.65:
                    trend_pct = 2
                elif no_trend_pct / len(pct) > 0.65:
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
        transmission_rate, awb_min, awb_max, grey_min, grey_max = calculate_parameters(trend_overall, transmission_rate,
                                                                                       awb_min, awb_max, grey_min,
                                                                                       grey_max)
        utility.print_verbose("New Range [{},{}]".format(awb_min, awb_max), verbose)
        transmission_interval = calculate_transmission_interval(transmission_rate, packet_size)
        # Terminate and return
        utility.print_verbose("Runtime for 1 fleet: {}s".format(timeit.default_timer() - start), verbose)
        loop_counter += 1
        time.sleep(2)
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


def calculate_parameters(trend, current_rate, rate_min, rate_max, grey_min, grey_max, resolution):
    """
    Adjust awb_estimate parameters according on the result.

    :param trend -- the trend of current iteration
    :param current_rate -- probing rate
    :param rate_min -- minimal awb bound
    :param rate_max -- maximal awb bound
    :param grey_min -- grey trend minimal awb bound
    :param grey_max -- grey trend maximal awb bound
    :param resolution -- step to probe
    """
    new_rate = 0
    if trend == 0:
        rate_min = current_rate
        new_rate = current_rate + (abs(rate_max - rate_min)) * resolution
    elif trend == 2:
        rate_max = current_rate
        new_rate = (rate_max + rate_min) / 2
    else:
        rate_max = rate_max - current_rate * resolution
        rate_min = rate_min + current_rate * resolution
        new_rate = current_rate + current_rate * resolution
    return new_rate, rate_min, rate_max, grey_min, grey_max


def calculate_transmission_interval(rate, packet_size):
    return packet_size / rate


def plot_results(packet_train_response, round_trip_times, filename='rtt.png', clear=False):
    mp.plot(np.array(range(len(packet_train_response))), np.array(round_trip_times), 'o')
    mp.ylabel("Round trip time in second")
    mp.xlabel("Time in seconds")
    mp.savefig('plots/' + filename, format='png')
    mp.show()
    if clear:
        mp.clf()


if __name__ == '__main__':
    estimate_available_bandwidth(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), True)
    # estimate_available_bandwidth('google.com', 5, 10, True)
