 import sys
 from scapy.all import *
 
 def ping(ip):
     print("Pinging target ...")
     icmp = IP(dst=ip)/ICMP()
     icmp.show()
     rsp = sr1(icmp, timeout=10)
     if rsp == None:
         print("Host is down")
     else:
         print("Host is up")
 
 
 def send_ack(ip, ack_number):
     print("Sending ACK...")
     ip = sys.argv[1]
     tcp_ack = IP(dst=ip)/TCP(ack=ack_number, flags = 'A')
     tcp_ack.show()
     rsp = sr1(tcp_ack)
     if rsp == None:
         print("Host is down or discarding")
     else:
         print("Host is responding")
         rsp.show()
 if __name__ == '__main__':
     ip = sys.argv[1]
     send_ack(ip, 0)