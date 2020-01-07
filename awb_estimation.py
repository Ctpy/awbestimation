import scapy_util
import pcap_util
import trend

global MIN_TRAIN_LENGTH


def estimate_available_bandwidth(target, capacity, resolution, tcpdumpfile, verbose):
    """
    Estimate the available bandwidth to the given target.
    Higher resolution will impact the performance.

    :param target -- IP of target to estimate
    :param capacity -- Bottleneck/Link capacity to the target
    :param resolution -- Accuracy of the estimation
    :param tcpdumpfile -- tcpdump logging file
    :param verbose -- more output
    """

    awb_min = None
    awb_max = None
    # Config Data here

    # In Mbits
    transmission_rate = capacity * 0.75
    # In Byte
    packet_size = 1500
    # Numbers of packets per train
    train_length = 100
    current_ack_number = 1
    last_ack_number = 1
    # Probe starts here
    for i in range(12):
        print("Currently running with these Parameters: ")

        # Send_fleet
        transmission_interval = calculate_transmission_interval(transmission_rate, train_length, packet_size)
        packet_train_numbers = generate_packet_train(current_ack_number, train_length)
        scapy_util.send_train(target, packet_train_numbers, transmission_interval, verbose)

        # Process pcap file and analyze csv file
        csv_file = tcpdumpfile.split('.')[0]
        pcap_util.convert_to_csv(tcpdumpfile, csv_file, packet_train_numbers)
        timestamps, packet_loss = pcap_util.analyze_csv(csv_file, packet_train_numbers)
        packet_loss_rate = packet_loss/train_length
        # calculate trend
        trend_state = trend.calculate_trend(timestamps, packet_loss, train_length)

        # fabprobe logic
        # TODO: Implement cases
        if trend_state.__eq__("INCREASE"):
            transmission_rate = None
        elif trend_state.__eq__("STABLE"):
            transmission_rate = None

        # Adjust state variables
        calculate_parameters()
    # Terminate and return

    return [awb_min, awb_max]


def generate_packet_train(starting_number, size):
    """
    Generate Acknowledgement numbers for a packet train.
    """

    train = []
    for i in range(size):
        train.append(1 + starting_number + 40 * i)
    return train


def calculate_parameters(results):
    parameters = []
    # TODO: Implement
    return parameters


def calculate_transmission_interval(transmission_rate, train_length, packet_size):
    return (train_length * packet_size)/transmission_rate


if __name__ == '__main__':
    estimate_available_bandwidth(100, 5)
