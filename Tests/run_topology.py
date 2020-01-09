from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.clean import cleanup
from subprocess import PIPE
# from mininet.cli import CLI

# import matplotlib.pyplot as plt
# import networkx as nx

import sys
import time


class CrossTrafficTopo(Topo):

    def build(self, capacities, size=1):
        """
        Declare topology components and links and build topology
        :param capacities: Link capacity distribution list
        :param size: Amount of intermediary switches
        """

        # At least one sender and receiver required
        assert size >= 1, "Topology size must be at least 1!"

        # Declare main sender and receiver host
        leftHost = self.addHost('leftHost')
        rightHost = self.addHost('rightHost')

        # Declare cross traffic sender and receiver host
        top_host = self.addHost('top_host')
        bottom_host = self.addHost('bottom_host')

        # Iterate through path using last node
        last_node = leftHost

        # Add switches
        for i in range(0, size):
            sw = self.addSwitch('sw' + str(i))
            # Left link
            self.addLink(sw, last_node)
            # Bottom Link
            self.addLink(bottom_host, sw)
            # Upper Link
            self.addLink(top_host, sw)
            last_node = sw

        # Connect last node with main receiver
        self.addLink(last_node, rightHost)


def build_topo(switch_count, duration, capacities, cross_traffic, verbose=False):
    """
    Build and execute topology
    :param switch_count: Amount of intermediary switches
    :param duration: Measurement duration in seconds
    :param capacities: Link capacity distribution list
    :param cross_traffic: Cross traffic load
    :param verbose: Verbose output mode
    """
    try:
        # Initialize topology
        topo = CrossTrafficTopo(capacities, size=switch_count)
        net = Mininet(topo=topo, link=TCLink)
        net.start()
        # if plot:
        #     plot_topo(topo)
    except Exception as e:
        print('Could not start Mininet!')
        print(e)
        sys.exit(1)

    graph = "[S]--{}--".format(capacities[0])
    for c in capacities[1:]:
        graph += "<x>--{}--".format(c)
    graph += "[R]"

    if verbose:
        print('Created topology with ' + str(switch_count) + ' switches.')
        print('Capacity distributions: {}'.format(graph))

    left_host = net.get('left_host')
    right_host = net.get('right_host')

    left_host.setIP('10.0.0.1/24')
    right_host.setIP('10.0.0.2/24')

    top_host = net.get('top_host')
    bottom_host = net.get('bottom_host')

    # Assign IP addresses
    for i in range(0, switch_count):
        top_host.cmd('ip a add 10.0.{}.1/24 dev top_host-eth{}'.format(i + 1, i))
        bottom_host.cmd('ip a add 10.0.{}.2/24 dev bottom_host-eth{}'.format(i + 1, i))

    # Remove IPs that were automatically assigned by Mininet
    top_host.cmd('ip a del 10.0.0.4/8 dev top_host-eth0')
    bottom_host.cmd('ip a del 10.0.0.1/8 dev bottom_host-eth0')

    # Assign new routing tables
    set_routing_tables(switch_count, top_host, bottom_host)

    # Set link capacities
    set_capacities(switch_count, capacities, net)

    left_host.cmd('tc qdisc replace dev left_host-eth0 root fq pacing')
    left_host.cmd('ethtool -K left_host-eth0 tso off')
    right_host.cmd('tc qdisc replace dev right_host-eth0 root netem delay 50')

    # CLI(net)

    right_host.cmd('iperf -s -t {} &'.format(duration + 2))

    try:
        right_host.popen('tcpdump -i right_host-eth0 tcp -w receiver.pcap', stdout=PIPE, stderr=PIPE)
        left_host.popen('tcpdump -i left_host-eth0 tcp -w sender.pcap', stdout=PIPE, stderr=PIPE)
        # Link logging
        # sw1 = net.get('sw1')
        # sw1.popen('tcpdump -i sw1-eth4  tcp -w sw1.pcap', stdout=PIPE, stderr=PIPE)
    except Exception as e:
        print('Error on starting tcpdump\n{}'.format(e))
        sys.exit(1)

    if verbose:
        print('Started tcpdump')

    # Wait for tcpdump to initialize
    time.sleep(1)

    # Start cross traffic connections
    if cross_traffic != 0:
        for i in range(2, switch_count + 1):
            # Receiver (logging: &>> receiver_log.txt)
            cmd = 'iperf -s -t {} -B 10.0.{}.2'.format(duration + 2, i)
            bottom_host.popen(cmd, stdout=PIPE, stderr=PIPE)

        for i in range(1, switch_count):
            # Sender (logging: &>> sender_log.txt)
            cmd = 'iperf -c 10.0.{}.2 -t {} -B 10.0.{}.1 -b {}M'.format(i + 1, duration + 2, i,
                                                                        capacities[i] * cross_traffic)
            top_host.popen(cmd, stdout=PIPE, stderr=PIPE)

        if verbose:
            print('Started cross traffic flows')

    try:
        if verbose:
            print('Running main file transfer...')
        # left_host.cmd('iperf -t {} -c {} &'.format(duration, right_host.IP()))
        left_host.cmd('sudo python test_send_receive.py {} {} {}'.format(right_host.IP(), 100, 0.1))
        time.sleep(duration + 1)
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print('\nReceived keyboard interrupt. Stop Mininet.')
        else:
            print(e)
    finally:
        if verbose:
            print('Done!')
        net.stop()
        cleanup()


def set_routing_tables(switch_count, top_host, bottom_host):
    """
    Define routing tables to correctly distribute cross traffic
    :param switch_count: Amount of intermediary switches
    :param top_host: Cross traffic sender host
    :param bottom_host: Cross traffic receiver host
    """
    # Clear tables
    top_host.cmd("ip r flush table main")
    bottom_host.cmd("ip r flush table main")

    # Add new entries
    for i in range(switch_count - 1):
        top_host.cmd("ip r add 10.0.{}.0/24 dev topHost-eth{} src 10.0.{}.1".format(i + 2, i, i + 1))
        bottom_host.cmd("ip r add 10.0.{}.0/24 dev bottomHost-eth{} src 10.0.{}.2".format(i + 1, i + 1, i + 2))


def set_capacities(switch_count, capacities, net):
    """
    Set link capacities by applying traffic limiters throughout the path
    :param switch_count: Amount of intermediary switches
    :param capacities: Link capacity distribution list
    :param net: Mininet network object
    """
    # Get main sender and receiver
    left_host = net.get('left_host')
    right_host = net.get('right_host')

    # Apply traffic limiters to first and last link
    left_host.cmd('tc qdisc add dev left_host-eth0 root netem rate {}mbit'.format(capacities[0]))
    right_host.cmd('tc qdisc add dev right_host-eth0 root netem rate {}mbit'.format(capacities[-1]))

    for i in range(0, switch_count):
        # Apply traffic limiter at switch i
        switch = net.get('sw{}'.format(i))
        # Link before switch
        switch.cmd(
            'tc qdisc add dev sw{}-eth1 root tbf rate {}mbit latency 100ms buffer 16000b'.format(i, capacities[i]))
        # Link after switch
        switch.cmd(
            'tc qdisc add dev sw{}-eth4 root tbf rate {}mbit latency 100ms buffer 16000b'.format(i, capacities[i + 1]))


def run_topo(capacities, duration, cross_traffic, verbose, **kwargs):
    """
    Run measurement
    :param capacities: Link capacity distribution list
    :param duration: Measurement duration in seconds
    :param cross_traffic: Cross traffic load
    :param verbose: Verbose output mode
    """
    switch_count = len(capacities) - 1

    try:
        build_topo(switch_count, duration, capacities, cross_traffic, verbose=verbose)
    except Exception as e:
        print(e)
        print('Cleaning up environment...')
        cleanup()
