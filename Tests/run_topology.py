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
        topHost = self.addHost('topHost')
        bottomHost = self.addHost('bottomHost')

        # Iterate through path using last node
        lastNode = leftHost

        # Add switches
        for i in range(0, size):
            sw = self.addSwitch('sw' + str(i))
            # Left link
            self.addLink(sw, lastNode)
            # Bottom Link
            self.addLink(bottomHost, sw)
            # Upper Link
            self.addLink(topHost, sw)
            lastNode = sw

        # Connect last node with main receiver
        self.addLink(lastNode, rightHost)


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

    leftHost = net.get('leftHost')
    rightHost = net.get('rightHost')

    leftHost.setIP('10.0.0.1/24')
    rightHost.setIP('10.0.0.2/24')

    topHost = net.get('topHost')
    bottomHost = net.get('bottomHost')

    # Assign IP addresses
    for i in range(0, switch_count):
        topHost.cmd('ip a add 10.0.{}.1/24 dev topHost-eth{}'.format(i + 1, i))
        bottomHost.cmd('ip a add 10.0.{}.2/24 dev bottomHost-eth{}'.format(i + 1, i))

    # Remove IPs that were automatically assigned by Mininet
    topHost.cmd('ip a del 10.0.0.4/8 dev topHost-eth0')
    bottomHost.cmd('ip a del 10.0.0.1/8 dev bottomHost-eth0')

    # Assign new routing tables
    set_routing_tables(switch_count, topHost, bottomHost)

    # Set link capacities
    set_capacities(switch_count, capacities, net)

    leftHost.cmd('tc qdisc replace dev leftHost-eth0 root fq pacing')
    leftHost.cmd('ethtool -K leftHost-eth0 tso off')
    rightHost.cmd('tc qdisc replace dev rightHost-eth0 root netem delay 50')

    # CLI(net)

    rightHost.cmd('iperf -s -t {} &'.format(duration + 2))

    try:
        rightHost.popen('tcpdump -i rightHost-eth0 tcp -w receiver.pcap', stdout=PIPE, stderr=PIPE)
        leftHost.popen('tcpdump -i leftHost-eth0 tcp -w sender.pcap', stdout=PIPE, stderr=PIPE)
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
            bottomHost.popen(cmd, stdout=PIPE, stderr=PIPE)

        for i in range(1, switch_count):
            # Sender (logging: &>> sender_log.txt)
            cmd = 'iperf -c 10.0.{}.2 -t {} -B 10.0.{}.1 -b {}M'.format(i + 1, duration + 2, i,
                                                                        capacities[i] * cross_traffic)
            topHost.popen(cmd, stdout=PIPE, stderr=PIPE)

        if verbose:
            print('Started cross traffic flows')

    try:
        if verbose:
            print('Running main file transfer...')
        leftHost.cmd('iperf -t {} -c {} &'.format(duration, rightHost.IP()))
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


def set_routing_tables(switch_count, topHost, bottomHost):
    """
    Define routing tables to correctly distribute cross traffic
    :param switch_count: Amount of intermediary switches
    :param topHost: Cross traffic sender host
    :param bottomHost: Cross traffic receiver host
    """
    # Clear tables
    topHost.cmd("ip r flush table main")
    bottomHost.cmd("ip r flush table main")

    # Add new entries
    for i in range(switch_count - 1):
        topHost.cmd("ip r add 10.0.{}.0/24 dev topHost-eth{} src 10.0.{}.1".format(i + 2, i, i + 1))
        bottomHost.cmd("ip r add 10.0.{}.0/24 dev bottomHost-eth{} src 10.0.{}.2".format(i + 1, i + 1, i + 2))


def set_capacities(switch_count, capacities, net):
    """
    Set link capacities by applying traffic limiters throughout the path
    :param switch_count: Amount of intermediary switches
    :param capacities: Link capacity distribution list
    :param net: Mininet network object
    """
    # Get main sender and receiver
    leftHost = net.get('leftHost')
    rightHost = net.get('rightHost')

    # Apply traffic limiters to first and last link
    leftHost.cmd('tc qdisc add dev leftHost-eth0 root netem rate {}mbit'.format(capacities[0]))
    rightHost.cmd('tc qdisc add dev rightHost-eth0 root netem rate {}mbit'.format(capacities[-1]))

    for i in range(0, switch_count):
        # Apply traffic limiter at switch i
        switch = net.get('sw{}'.format(i))
        # Link before switch
        switch.cmd(
            'tc qdisc add dev sw{}-eth1 root tbf rate {}mbit latency 100ms buffer 16000b'.format(i, capacities[i]))
        # Link after switch
        switch.cmd(
            'tc qdisc add dev sw{}-eth4 root tbf rate {}mbit latency 100ms buffer 16000b'.format(i, capacities[i + 1]))


# def plot_topo(topo):
#     """
#     Create and show topology plot
#     :param topo: topology object
#     """
#     G = nx.Graph()
#     for n in topo.g.node:
#         G.add_node(n)
#     for n1 in topo.g.edge:
#         for n2 in topo.g.edge[n1]:
#            G.add_edge(n1, n2)
#
#     nx.draw(G, with_labels=True)
#     print(G)
#     plt.show()


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
