import random
import time
import json


def get_capacity_distribution(capacity_range, capacity_delta, switch_count):
    """
    Create a capacity distribution list to be applied on the measurement path
    :param capacity_range: Two element list representing range to select capacities from: [minimum, maximum]
    :param capacity_delta: Defines steps for capacity range
    :param switch_count: Amount of intermediary switches throughout the path
    :return: List consisting of generated link capacities
    """
    assert not capacity_delta <= 0, "capacity_delta must be a positive value!"

    # First link between main sender and first switch: use maximum
    ret = [max(capacity_range)]

    for i in range(switch_count - 1):
        try:
            # Select random integer using given parameters
            capacity = random.randrange(capacity_range[0], capacity_range[1], capacity_delta)
        except ValueError:
            capacity = capacity_range[0]
        # Add generated link capacity to distribution
        ret.append(capacity)

    # Last link between last switch and main receiver: use maximum
    ret.append(max(capacity_range))

    return ret


def parse_config(args):
    """
    Process json configuration file and create parameter dict for test topology
    :param args: ArgumentParser object passed from main function
    :return: dict containing parameters for testing topology
    """
    # Result folder structure: config filename + current time stamp
    ret = {'folder_name': args.config.replace('.json', '') + str(int(time.time()))}

    # Read json file
    try:
        with open(args.config) as json_file:
            data = json.load(json_file)
    except ValueError:
        print('Invalid input json file!')

    # Read capacity distribution range
    capacity_range = data['capacity_range']
    assert len(capacity_range) == 2, "Capacity range must be an interval of two numbers!"

    # Read capacity distribution step value
    capacity_delta = data['capacity_delta']
    assert capacity_delta > 0, "Capacity delta must be a positive number!"

    # Read measurement duration in seconds
    duration = data['duration']
    assert duration > 0, "Duration must be a positive number!"
    ret['duration'] = data['duration']

    # Read intermediary switch count
    switch_count = data['switch_count']
    assert switch_count > 0, "Switch count must be a positive number!"
    ret['switch_count'] = switch_count

    # Read amount of cross traffic load
    try:
        ret['cross_traffic'] = data['cross_traffic']
    except KeyError:
        # No parameter given, assume cross traffic load = 1
        ret['cross_traffic'] = 1

    # Read optional packet size
    try:
        ret['packet_size'] = data['packet_size']
    except KeyError:
        # No parameter given, use default Ethernet + IP + MSS size of 1515
        ret['packet_size'] = 1515

    # Read optional bottleneck location
    try:
        bottleneck_location = data['bottleneck_location']
        if bottleneck_location >= switch_count:
            print("Bottleneck location out of link bounds! (Ignoring parameter)")
            bottleneck_location = None
    except KeyError:
        bottleneck_location = None

    # Declare verbose execution
    try:
        ret['verbose'] = data['verbose']
    except KeyError:
        ret['verbose'] = True

    # Decide whether to keep log files
    try:
        ret['keep_log'] = data['keep_log']
    except KeyError:
        ret['keep_log'] = True

    # Calculate random capacities for all links
    if bottleneck_location is None:
        # Random bottleneck location
        ret['capacities'] = get_capacity_distribution(capacity_range, capacity_delta, switch_count)
    else:
        # Defined bottleneck location
        ret['capacities'] = get_capacity_distribution([capacity_range[0] + capacity_delta, capacity_range[1]],
                                                      capacity_delta, switch_count)
        ret['capacities'][bottleneck_location] = capacity_range[0]
    # Set bottleneck value to minimum capacity value in link distribution
    ret['bottleneck'] = min(ret['capacities'])

    # Declare optional output file to append results to
    try:
        ret['output'] = data['output']
    except KeyError:
        ret['output'] = None

    # Return dict
    return ret
