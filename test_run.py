 from mininet.topo import Topo
 from mininet.net import Mininet
 from threading import Thread
 
 class TrafficTopology(Topo):
 
     def build(self):
         leftHost = self.addHost('h1')
         rightHost = self.addHost('h2')
         sw = self.addSwitch('s1')
         self.addLink(leftHost, sw)
         self.addLink(sw, rightHost)
 
 def build_topo():
     try:
         topo = TrafficTopology()
         net = Mininet(topo=topo)
         net.start()
     except Exception as e:
         print(e)
         print('Cleaning up environment...')
         cleanup()
     net.pingAll()
     threads = []
     h1 = net.get('h1')
     h2 = net.get('h2')
     h1.setIP('10.0.0.1/24')
     h2.setIP('10.0.0.2/24')
     print(h2.cmdPrint('sudo python sniff.py &'))
     print(h1.cmdPrint('sudo python scapy_util.py %s' % h2.IP() ))
     net.stop()
 
 if __name__ == '__main__':
     build_topo()