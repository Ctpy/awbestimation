#include <stdio.h>	//for printf
#include <string.h> //memset
#include <sys/socket.h>	//for socket ofcourse
#include <stdlib.h> //for exit(0);
#include <errno.h> //For errno - the error number
#include <netinet/tcp.h>	//Provides declarations for tcp header
#include <netinet/ip.h>	//Provides declarations for ip header
#include <unistd.h>
#include <arpa/inet.h>

struct pseudo_header
{
	u_int32_t source_address;
	u_int32_t dest_address;
	u_int8_t placeholder;
	u_int8_t protocol;
	u_int16_t tcp_length;
};

unsigned short csum(unsigned short *ptr,int nbytes)
{
	register long sum;
	unsigned short oddbyte;
	register short answer;

	sum=0;
	while(nbytes>1) {
		sum+=*ptr++;
		nbytes-=2;
	}
	if(nbytes==1) {
		oddbyte=0;
		*((u_char*)&oddbyte)=*(u_char*)ptr;
		sum+=oddbyte;
	}

	sum = (sum>>16)+(sum & 0xffff);
	sum = sum + (sum>>16);
	answer=(short)~sum;

	return(answer);
}

int create_tcp_ack(char *packet, int ack_number, const char *dst, size_t packet_size){
    // Create packet array
    int header_size = 52 * 8; // in bit ?
    struct pseudo_header psh;
    char *data, *pseudogram;
    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
	sin.sin_port = htons(80);
	sin.sin_addr.s_addr = inet_addr (dst);
    // Init packer buffer
    printf("Init buffer\n");
    printf("size packet: %i, packet_size: %i\n", sizeof(packet), packet_size);
    memset(packet, 0, packet_size);
    printf("memset\n");
    struct iphdr *ip = (struct iphdr *) packet;
    printf("Init ip\n");
    struct tcphdr *tcp = (struct tcphdr *) (packet + sizeof (struct iphdr));
    printf("Init tcp\n");
    data = packet + sizeof(struct iphdr) + sizeof(struct tcphdr);
    printf("Init data\n");
    memset(data, 'x', packet_size - sizeof(struct iphdr) + sizeof(struct tcphdr));
    printf("Init data mem\n");
    // Create IP Header
    ip->frag_off = 0;
    printf("Init frag\n");
    ip->version = 4;
    ip->ihl = 5;
    printf("Init frag ver ihl\n");
    ip->tot_len = htons(sizeof(struct iphdr) + sizeof(struct tcphdr));
    ip->id = 0;
    ip->ttl = 64;
    ip->protocol = IPPROTO_TCP;
    printf("Init proto\n");
    ip->saddr = INADDR_ANY; // Kernel will specify it
    ip->daddr = sin.sin_addr.s_addr;
    ip->check = 0; 
    printf("Not csum yet\n");
    ip->check = csum((unsigned short*) packet, ip->tot_len);
    printf("Filled IP header\n");
    // Create TCP Header
    tcp->source     = htons(1234);
    tcp->dest       = htons(80);
    tcp->seq        = 0;
    tcp->ack_seq   = ack_number;
    tcp->doff       = 5;
    tcp->ack        = 1;
    tcp->psh        = 0;
    tcp->rst        = 0;
    tcp->urg        = 0;
    tcp->syn        = 0;
    tcp->fin        = 0;
    tcp->window     = htons(65535);
    
    psh.source_address = ip->saddr;
    psh.dest_address = ip->daddr;
    psh.placeholder = 0;
    psh.protocol = ip->protocol;
    psh.tcp_length = htons(sizeof(struct tcphdr) + strlen(data));
    
    int psize = sizeof(struct pseudo_header) + sizeof(struct tcphdr) + strlen(data);
    pseudogram = malloc(psize);

    memcpy(pseudogram , (char*) &psh , sizeof (struct pseudo_header));
	memcpy(pseudogram + sizeof(struct pseudo_header) , tcp , sizeof(struct tcphdr) + strlen(data));

    tcp->check = csum( (unsigned short*) pseudogram , psize);
    
    return ip->tot_len;
}

int send_packet_train(int *array, size_t size, const char *dst, int time_interval, size_t packet_size){
    // Create array of packets -> 2D array
    char packet_train[size][packet_size * 8];
    int sock, bytes, on = 1;
    struct sockaddr_in addr;
    printf("Initialized\n");
    // Create packets and store in array
    int tot_len;
    for(int i = 0; i < size; i++){
        tot_len = create_tcp_ack(packet_train[i], array[i], dst, packet_size * 8);
    }
    printf("Created packets\n");
    // create RAW Socket
    sock = socket(AF_INET, SOCK_RAW, IPPROTO_TCP);
    if (sock == -1) {
        perror("Socket() failed");
        return 1;
    }else{
        printf("Socket() sucessful\n");
    }
    if (setsockopt(sock, IPPROTO_IP, IP_HDRINCL, &on, sizeof(on)) == -1) {
        perror("setsockopt() failed");
        return 2;
    }else{
        printf("setsockopt() ok\n");
    }
    addr.sin_addr.s_addr = inet_addr(dst);
    addr.sin_family = AF_INET;
    addr.sin_port = htons(80);
    // Send in loop with timeinterval
    for(int i=0; i < size; i++){
        if(sendto(sock, packet_train[i], tot_len, 0, (struct sockaddr*) &addr, sizeof(addr))){
            perror("Sending failed\n");
            return 1;
        } else {
            printf ("Packet Send. Length : %d \n" , tot_len);
        }
        usleep(time_interval);
    }
    return 0;
}
int main(){
    int ack[1];
    ack[0] = 22;
    printf("Start\n");
    send_packet_train(ack, 1, "google.com", 100, 1500);
    
}
