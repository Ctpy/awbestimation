import time
import scapy_util
import pcap_util
import trend


def estimate_available_bandwidth(target, capacity, resolution, tcpdump_file, verbose):
    """
    Estimate the available bandwidth to the given target.
    Higher resolution will impact the performance.

    :param target -- IP of target to estimate
    :param capacity -- Bottleneck/Link capacity to the target
    :param resolution -- Accuracy of the estimation
    :param tcpdump_file -- tcpdump logging file
    :param verbose -- more output
    """

    # Config Data here
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
    # Probe starts here
    for i in range(12):
        print("Currently running with these Parameters: ")

        # Send_fleet
        transmission_interval = calculate_transmission_interval(transmission_rate, train_length, packet_size)
        packet_train_numbers = generate_packet_train(current_ack_number, train_length)
        last_ack_number = packet_train_numbers[-1] + 40
        scapy_util.send_train(target, packet_train_numbers, transmission_interval, verbose)

        # Process pcap file and analyze csv file
        csv_file = tcpdump_file.split('.')[0]
        pcap_util.convert_to_csv(tcpdump_file, csv_file, packet_train_numbers)
        timestamps, packet_loss = pcap_util.analyze_csv(csv_file, packet_train_numbers)
        packet_loss_rate = packet_loss/train_length

        # calculate trend
        trend_state, pct_trend, pdt_trend = trend.calculate_trend(timestamps, packet_loss, train_length)
        trend_list.append(trend_state)
        pct_trend_list.append(pct_trend)
        pdt_trend_list.append(pdt_trend)

        # fabprobe logic
        if i > 1:
            if (trend_list[-1] == "NOCHANGE" or trend_list[-1] == "UNCL") and trend_list[-2] == "INCR":
                return awb_min, awb_max

        # Adjust state variables
        current_awb, train_length, transmission_interval = calculate_parameters(trend_state, train_length, transmission_interval, awb_min, awb_max, packet_loss_rate, packet_size)
        if current_awb > awb_max:
            awb_max = current_awb
        elif current_awb < awb_min:
            awb_min = current_awb
        current_ack_number = last_ack_number
        # wait that fleets dont interfere
        time.sleep(1)
    # Terminate and return

    return awb_min, awb_max


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
    return (train_length * packet_size)/transmission_rate
