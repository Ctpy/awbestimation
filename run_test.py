import os
import argparse
import prepare_test
import run_topology
import numpy as np


def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to the config file')
    args = parser.parse_args()

    # Get parameters from json input and prepare test
    test_parameters = prepare_test.parse_config(args)
    bottleneck = min(test_parameters['capacities'])

    # Run setup using input parameters
    run_topology.run_topo(**test_parameters)

    # # Convert pcap files to csv
    # if test_parameters['verbose']:
    #     print("Converting capture files...")
    # os.system("mkdir {}".format(test_parameters['folder_name']))
    # analyze_csv.convert_pcap(test_parameters['folder_name'], 'receiver.pcap')
    # analyze_csv.convert_pcap(test_parameters['folder_name'], 'sender.pcap')
    # os.system("rm receiver.pcap sender.pcap")
    #
    # # Estimate capacities and save log
    # if test_parameters['verbose']:
    #     print("Estimating capacity...")
    # results = analyze_csv.get_results(test_parameters['folder_name'], test_parameters['bottleneck'],
    #                                   test_parameters['capacities'], size=test_parameters['packet_size'])
    # log_output = open('./{}/log.txt'.format(test_parameters['folder_name']), 'r').read()
    #
    # if test_parameters['verbose']:
    #     print(log_output)
    #
    # if test_parameters['output'] is not None:
    #     analyze_csv.write_result_csv(test_parameters['output'], test_parameters['bottleneck'] * 10 ** 6, results)
    #
    # if test_parameters['keep_log'] is False:
    #     os.system("rm -r ./{}".format(test_parameters['folder_name']))


def cross_traffic_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cmin', dest='capacity_min', help='Capacity Range Minimum')
    parser.add_argument('-cmax', dest='capacity_max', help='Capacity Range Maximum')
    parser.add_argument('-ct', dest='ctraffic_amount', help='Cross Traffic Amount')
    parser.add_argument('-cd', dest='capacity_delta', help='Capacity Delta')
    parser.add_argument('-s', dest='switch_count', help='Switch Count')
    parser.add_argument('-d', dest='duration', help='Duration')
    parser.add_argument('-o', dest='output', help='Output File')
    args = parser.parse_args()

    capacity_range = [int(args.capacity_min), int(args.capacity_max)]
    capacity_delta = int(args.capacity_delta)
    switch_count = int(args.switch_count)
    output = args.output
    ctraffic_amount = float(args.ctraffic_amount)
    duration = int(args.duration)
    capacities = prepare_test.get_capacity_distribution(capacity_range, capacity_delta, switch_count)
    bottleneck = min(capacities)
    estimates = [[], [], []]

    for i in range(10):
        parameters = {'folder_name': 'ct_test', 'capacities': capacities,
                      'duration': duration, 'cross_traffic': ctraffic_amount,
                      'bottleneck': bottleneck, 'switch_count': switch_count}

        run_topology.run_topo(**parameters)
        os.system("mkdir {}".format(parameters['folder_name']))
        analyze_csv.convert_pcap(parameters['folder_name'], 'receiver.pcap')
        analyze_csv.convert_pcap(parameters['folder_name'], 'sender.pcap')
        os.system("rm receiver.pcap sender.pcap")

        estimates[0].append(
            analyze_csv.get_capacity("{}/receiver.csv".format(parameters['folder_name']), "10.0.0.1", "10.0.0.2",
                                     receiver_mode=True))
        estimates[1].append(
            analyze_csv.get_capacity("{}/receiver.csv".format(parameters['folder_name']), "10.0.0.1", "10.0.0.2",
                                     receiver_mode=False))

        estimates[2].append(
            analyze_csv.get_capacity("{}/sender.csv".format(parameters['folder_name']), "10.0.0.1", "10.0.0.2",
                                     receiver_mode=False))

        os.system("rm -r ct_test")

    with open(output, 'a') as results:
        means = [np.mean(estimates[0]), np.mean(estimates[1]), np.mean(estimates[2])]
        stdevs = [np.std(estimates[0], ddof=1), np.std(estimates[1], ddof=1), np.std(estimates[2], ddof=1)]

        results.write(
            str(ctraffic_amount) + ';' + str(means[0] - stdevs[0]) + ';' + str(means[0]) + ';' + str(
                means[0] + stdevs[0]) + ';' +
            str(means[1] - stdevs[1]) + ';' + str(means[1]) + ';' + str(means[1] + stdevs[1]) + ';' +
            str(means[2] - stdevs[2]) + ';' + str(means[2]) + ';' + str(means[2] + stdevs[2]) + '\n')


if __name__ == '__main__':
    main()
    # cross_traffic_test()
