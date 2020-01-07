from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.clean import cleanup
from subprocess import sp
import sys
import time


class BottleneckTopo(Topo):

    def build(self, capacities):
        # Add hosts
        left_upper_host = self.addHost('leftUpperHost')
        left_lower_host = self.addHost('leftLowerHost')
        right_upper_host = self.addHost('rightUpperHost')
        right_lower_host = self.addHost('rightLowerHost')

        # Add switches
        # TODO: Add switches and assign capacities accordingly

        sw1 = self.addSwitch('sw1')
        self.addLink(sw1, left_lower_host, bw=1)
        self.addLink(sw1, left_upper_host, bw=1)
        sw2 = self.addSwitch('sw2')
        self.addLink(sw2, right_lower_host, bw=1)
        self.addLink(sw2, right_upper_host, bw=1)
        self.addLink(sw1, sw2, bw=0.5)


# TODO make it adaptable by input
def build_topo(capacities: list):
    try:
        topo = BottleneckTopo(capacities)
        net = Mininet(topo=topo, link=TCLink)
        net.start()
    except Exception as e:
        # print(e)
        sys.exit(1)

    left_upper_host = net.get('leftUpperHost')
    left_lower_host = net.get('leftLowerHost')
    right_upper_host = net.get('rightUpperHost')
    right_lower_host = net.get('rightLowerHost')

    left_upper_host.setIP('10.0.0.1/24')
    left_lower_host.setIP('10.0.0.2/24')
    right_upper_host.setIP('10.0.0.3/24')
    right_lower_host.setIP('10.0.0.4/24')

    # Running stuff here

    # Running cross_traffic
    capacity = min(capacities)
    # left_lower host server side
    cmd = 'iperf -s &'
    print(left_lower_host.popen(cmd))
    left_lower_host.popen('tcpdump -t -w sender.pcap')
    # right_upper host client side
    cmd = 'iperf -c 10.0.0.2 -b 1M'
    print(right_upper_host.popen(cmd))
    # Clean up
    time.sleep(1)
    net.stop()
    cleanup()


if __name__ == '__main__':
    build_topo(10)
