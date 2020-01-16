from datetime import datetime
import subprocess
import pandas as pd


def convert_to_csv(filename, outputfile, packet_train):
    template = "tcp.ack=={} or tcp.seq=={}"
    filter_string = ""
    for i, packet in enumerate(packet_train):
        if i > 0:
            filter_string += " or "
        filter_string += template.format(packet, packet)
    print(filter)
    print("Converting pcap to csv format...")
    cmd = """tshark -r {} -T fields -e frame.number -e _ws.col.Time -e eth.src -e eth.dst -e ip.src -e ip.dst -e ip.proto -e tcp.ack -e tcp.flags -e tcp.seq -o tcp.relative_sequence_numbers:FALSE -E header=y -E separator=, -E quote=d -E occurrence=f -Y "{}" > {}""".format(
        filename, filter_string, outputfile)
    try:
        res = subprocess.check_output(['bash', '-c', cmd])
    except subprocess.CalledProcessError as e:
        res = e.output
    print(res)


def analyze_csv(file, id_list):
    # TODO: Re implement
    print("Analyse csv file...")
    data = pd.read_csv(file)
    timestamps = []
    average = 0
    packet_loss = 0
    arrival = []
    departure = []
    for i in id_list:
        # print("currently analyzing " + str(i))
        try:
            pair = (data[data['tcp.seq'] == i], data[data['tcp.ack'] == i])
            time_frame = str(pair[0]['frame.time'])
            # print("Frame time1" + time)
            time1 = time_frame.split(' ')
            new_time = [time1[4], time1[6], time1[7], time1[8]]
            time_frame = str(pair[1]['frame.time'])
            time2 = time_frame.split(' ')
            # print("Time 1:" + str(time1))
            # print(new_time)
            # print("Time 2:" + str(time2))
            new_time2 = [time2[4], time2[6], time2[7], time2[8]]
            # print(new_time2)
            # print(str(new_time[3])[:14])
            # print(str(new_time2[3])[:14])
            departure.append(new_time2)
            arrival.append(new_time)
            d1 = datetime.strptime(str(new_time[3])[:14], "%H:%M:%S.%f")
            d2 = datetime.strptime(str(new_time2[3])[:14], "%H:%M:%S.%f")
            result = abs(d2 - d1)
            # if(i == 0):
            #    pairs.append(new_time)
            # if(i == len(id_list) - 1):
            #    pairs.append(new_time2)
            # print((new_time2,new_time))
            # print("RTT " + str(i) + ": " + str(result))
            timestamps.append([str(new_time2[3])[:14], str(new_time[3])[:14], result.microseconds])
            average += result.microseconds
        except:
            packet_loss += 1

    print("Packets dropped: " + str(packet_loss))
    try:
        average /= (len(id_list) - packet_loss)
    except:
        average = None
    print("Average Time: " + str(average))
    # print(departure[0])
    # print(arrival[len(id_list) - 1 - packet_loss])
    return timestamps, packet_loss
