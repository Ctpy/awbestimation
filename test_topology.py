from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.clean import cleanup
# from subprocess import sp
import sys
import time


class BottleneckTopo(Topo):

    def build(self, capacities):
        # Add hosts
        left_host = self.addHost('leftHost')
        right_host = self.addHost('rightHost')

        # Add switches
        # TODO: Add switches and assign capacities accordingly

        sw1 = self.addSwitch('sw1')
        self.addLink(left_host, sw1, bw=0.5)
        self.addLink(sw1, right_host, bw=0.5)


# TODO make it adaptable by input
def build_topo(capacities):
    try:
        topo = BottleneckTopo(capacities)
        net = Mininet(topo=topo, link=TCLink)
        net.start()
        print("Build")
    except Exception as e:
        # print(e)
        sys.exit(1)

    left_host = net.get('leftHost')
    right_host = net.get('rightHost')

    left_host.setIP('10.0.0.1/24')
    right_host.setIP('10.0.0.2/24')

    # Running cross_traffic
    # capacity = min(capacities)
    # left_lower host server side
    cmd = 'iperf -s &'
    print(left_host.popen(cmd))
    print(left_host.popen('sudo python awb_estimation.py 10.0.0.2 0.5 10.0'))
    time.sleep(1)
    print("tcpdump")
    # right_upper host client side
    cmd = 'iperf -c 10.0.0.1 -b 1M'
    print(right_host.popen(cmd))
    # Clean up
    time.sleep(10)
    net.stop()
    cleanup()
    print("Clean up")


if __name__ == '__main__':
    build_topo(10)
