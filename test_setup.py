import json
import math
import subprocess
from random import random
import run_test
import create_test
from argparse import Namespace

def run_test_environment(test_config):
    # read the test specification
    data = None
    with open(test_config, 'r') as f:
        data = json.load(f)
    # create test_case
    iteration = data['iterations']
    cross_traffic_default = data['cross_traffic']
    cross_traffic_delta = data['cross_traffic_delta']
    rate_range = data['rate_range']
    switch_range = data['switch_range']
    test_config, bottleneck = create_test.create_test_case(rate_range[0], rate_range[1], switch_range[0],
                                                           switch_range[1], cross_traffic_default)
    # loop - test_specification
    for i in range(iteration):
        # run test
        print("Run test: " + test_config)
        run_test.main(Namespace(config=test_config))
        # evaluate test
        result = None
        with open('result.json', 'r') as f:
            result = json.load(f)
        true_available_bandwidth = bottleneck - bottleneck * cross_traffic_default
        estimated_available_bandwidth = result['result']
        if estimated_available_bandwidth[0] < true_available_bandwidth < estimated_available_bandwidth[1]:
            # reward
            absolute_error = 0
            relative_error = 0
            break
        else:
            absolute_error = min(math.fabs(estimated_available_bandwidth[0] - true_available_bandwidth),
                                 math.fabs(estimated_available_bandwidth[1] - true_available_bandwidth))
            relative_error = min(math.fabs(1 - estimated_available_bandwidth[0] / true_available_bandwidth),
                                 math.fabs(1 - estimated_available_bandwidth[1] / true_available_bandwidth))
        print("Relative Error" + str(relative_error))
        print("Absolute Error" + str(absolute_error))
    # loop - tweak cross traffic
    # for i in range(iteration):
    #     random_traffic = cross_traffic_default + random.uniform(cross_traffic_delta[0], cross_traffic_delta[1])
    #     test_config, bottleneck = create_test.create_test_case(rate_range[0], rate_range[1], switch_range[0],
    #                                                            switch_range[1], random_traffic)
    #     # run test
    #     subprocess.call(['sudo', 'python', 'run_test.py', test_config])
    #     # evaluate test
    #     true_available_bandwidth = bottleneck - bottleneck * random_traffic
    #     estimated_available_bandwidth = result['result']
    #     if estimated_available_bandwidth[0] < true_available_bandwidth < estimated_available_bandwidth[1]:
    #         # reward
    #         absolute_error = 0
    #         relative_error = 0
    #         break
    #     else:
    #         absolute_error = min(math.fabs(estimated_available_bandwidth[0] - true_available_bandwidth),
    #                              math.fabs(estimated_available_bandwidth[1] - true_available_bandwidth))
    #         relative_error = min(math.fabs(1 - estimated_available_bandwidth[0] / true_available_bandwidth),
    #                              math.fabs(1 - estimated_available_bandwidth[1] / true_available_bandwidth))

    # save results


if __name__ == '__main__':
    run_test_environment("package.json")
