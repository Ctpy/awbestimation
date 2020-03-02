import json
import math
import pandas
import numpy as np


def evaluate_result(result_file, bottleneck, cross_traffic_delta):
    # Evaluation criteria
    accuracy_improved = False
    performance_improved = False
    overhead_improved = False
    stable = False
    result = None
    with open('result.json', 'r') as f:
        result = json.load(f)
    true_available_bandwidth = bottleneck - bottleneck * cross_traffic_delta
    estimated_available_bandwidth = result['result']
    iteration_times = result['iteration_times']
    fleet_times = result['fleet_times']
    # Criteria 1: Accuracy - Errors
    in_scope = False
    if estimated_available_bandwidth[0] < true_available_bandwidth < estimated_available_bandwidth[1]:
        # reward
        in_scope = True
        absolute_error = max(math.fabs(estimated_available_bandwidth[0] - true_available_bandwidth),
                             math.fabs(estimated_available_bandwidth[1] - true_available_bandwidth))
        relative_error = max(math.fabs(1 - estimated_available_bandwidth[0] / true_available_bandwidth),
                             math.fabs(1 - estimated_available_bandwidth[1] / true_available_bandwidth))
    else:
        absolute_error = min(math.fabs(estimated_available_bandwidth[0] - true_available_bandwidth),
                             math.fabs(estimated_available_bandwidth[1] - true_available_bandwidth))
        relative_error = min(math.fabs(1 - estimated_available_bandwidth[0] / true_available_bandwidth),
                             math.fabs(1 - estimated_available_bandwidth[1] / true_available_bandwidth))
    print("In scope: " + in_scope)
    print("Relative Error" + str(relative_error))
    print("Absolute Error" + str(absolute_error))

    # Criteria 2: Performance - Track times - Resolution
    standard_deviation = np.mean(iteration_times)
    
    # Criteria 3: Overhead - Track log files

    # Criteria 4: Stability - Errors?

    # Evaluate using old knowledge
    # TODO: Store all results in a csv
    return None
