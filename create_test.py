import random
import json


def create_test_case(range_min, range_max, switch_count_min, switch_count_max, traffic):
    x = random.randint(range_min, range_max)
    y = random.randint(range_min, range_max)
    capacity_range = [min(x, y), max(x, y)]
    switch_count = random.randint(switch_count_min, switch_count_max)
    cross_traffic = traffic
    output = "result.csv"

    # Build json object
    result = "test-files/{}-{}-{}.json".format(switch_count, cross_traffic, random.random())
    data = {'capacity_range': capacity_range,
            'capacity_delta': 1,
            'duration': 180,
            'cross_traffic': cross_traffic,
            'verbose': True,
            'output': output
            }
    with open(result, 'w') as f:
        json.dump(data, f)
    return result, min(capacity_range)


if __name__ == '__main__':
    create_test_case(1, 10, 1, 3, 5, 1)