from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.clean import cleanup
from subprocess import sp
import sys
import time


class BottleneckTopo(Topo):

    def build(self, capacities, size):
        # Add hosts
        left_upper_host = self.addHost('leftUpperHost')
        left_lower_host = self.addHost('leftLowerHost')
        right_upper_host = self.addHost('rightUpperHost')
        left_lower_host = self.addHost('rightLowerHost')

        # Add switches
        # TODO: Add switches and assign capacities accordingly

    def run(self):
        # TODO: Create a testbed
        return
