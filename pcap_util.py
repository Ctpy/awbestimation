import subprocess
import pandas as pd


def convert_to_csv(filename, outputfile, packet_train):
    template = "tcp.ack=={} or tcp.seq=={}"
    filter_string = ""
    for i, packet in enumerate(packet_train):
        if i > 0:
            filter_string += " or "
        filter_string += template.format(packet, packet)
    print("Converting pcap to csv format...")
    cmd = """tshark -r {} -T fields -e frame.number -e _ws.col.Time -e eth.src -e eth.dst -e ip.src -e ip.dst -e ip.proto -e tcp.ack -e tcp.flags -e tcp.seq -o tcp.relative_sequence_numbers:FALSE -E header=y -E separator=, -E quote=d -E occurrence=f -Y "{}" > {}""".format(
        filename, filter_string, outputfile)
    try:
        res = subprocess.check_output(['bash', '-c', cmd])
    except subprocess.CalledProcessError as e:
        res = e.output
    print(res)


def analyze_csv(input_file, id_list):
    print("Analyse csv file...")
    data = pd.read_csv(input_file)
    timestamps = []
    unanswered = []
    packet_loss = 0
    rtt_sum = 0
    for i in id_list:
        try:
            pair = (data[data['tcp.ack'] == i], data[data['tcp.seq'] == i])
            time_frame_sent = pair[0]['_ws.col.Time'].item()
            time_frame_received = pair[1]['_ws.col.Time'].item()
            round_trip_time = abs(time_frame_received - time_frame_sent)
            timestamp = (time_frame_sent, round_trip_time)
            timestamps.append(timestamp)
            rtt_sum += round_trip_time
        except:
            frame = data[data['tcp.ack'] == i]
            time_frame_sent = frame['_ws.col.Time'].item()
            timestamp = (time_frame_sent, None)
            unanswered.append(timestamp)
            packet_loss += 1

    return timestamps,rtt_sum/(len(id_list)-packet_loss), packet_loss
