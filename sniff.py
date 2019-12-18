 from scapy.all import *
 
 print("started sniffing...")
 p = sniff(count=1,filter="10.0.0.1", prn=lambda x: x.show())
 print(p)
 
 print("end sniffing")