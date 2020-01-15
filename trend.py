import numpy as np
import math
import globals
import matplotlib.pyplot as mp


def calculate_trend(timestamps, packet_loss, train_length):
    np.set_printoptions(suppress=True)
    # Evaluate timestamps
    # TODO
    # Decreasing Trend due to buffering DT filter to filter out large bursts

    # Robust Regression filter to filter out small bursts by using Iteratively Re-weighted Least Square method (IRLS)

    # Evaluate trend
    return


def pct_metric(timestamps, packet_loss, train_length):
    increase = 0
    if train_length - packet_loss >= globals.MIN_TRAIN_LENGTH:
        for i in range(train_length - 1):
            if timestamps[i][2] < timestamps[i + 1][2]:
                increase += 1
        return increase / (train_length - packet_loss)
    else:
        return -1


def pdt_metric(timestamps, packet_loss, train_length):
    pdt_average = 0
    pdt_sum = 0
    if train_length - packet_loss >= globals.MIN_TRAIN_LENGTH:
        for i in range(train_length - 1):
            if timestamps[i] is None or timestamps[i + 1] is None:
                continue
            else:
                pdt_average += timestamps[i][2] - timestamps[i + 1][2]
                pdt_sum += abs(timestamps[i][2] - timestamps[i + 1][2])
        return pdt_average / pdt_sum
    return -1


def decreasing_trend_filter(timestamps):

    # search for burst
    mean = np.mean(timestamps)
    standard_derivation = np.std(timestamps)
    burst_packet_index_list = []
    print("Mean: " + str(mean))
    print("Standard Derivation: " + str(standard_derivation))
    for i in range(len(timestamps)):
        if mean + standard_derivation < timestamps[i]:
            burst_packet_index_list.append(i)
    print("Burst Index")
    print(burst_packet_index_list)
    # search for consecutive packet sample with decreasing rtt
    decreasing_trend_index_list = []
    for i in range(len(burst_packet_index_list)):
        decreasing_trend = True
        tmp = []
        last_packet_rtt = timestamps[burst_packet_index_list[i]]
        for j in range(1, 1 + globals.DT_CONSECUTIVE):
            if last_packet_rtt < timestamps[burst_packet_index_list[i] + j]:
                decreasing_trend = False
                break
            tmp.append(burst_packet_index_list[i] + j)
        if decreasing_trend:
            decreasing_trend_index_list.extend(tmp)
    print(decreasing_trend_index_list)
    burst_packet_index_list.append(decreasing_trend_index_list)
    print(type(burst_packet_index_list))

    list.sort(burst_packet_index_list)
    timestamps_np = np.array(timestamps)
    timestamps_np = np.delete(timestamps_np, decreasing_trend_index_list)

    return timestamps_np.tolist()


def robust_regression_filter(timestamps, packet_loss, train_length):
    iter_limit = 10
    columns = 2
    # TODO
    # OLS

    # return trend

    return


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    x = list(map(float,
                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                  28,
                  29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                  54,
                  55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                  80,
                  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]))
    # y = [-1, 1856, 604, 1371, 1724, 0, 1873, 1456, 863, 537, 2418, 1215, 1012, 606, 1502, 1195, 1497, 2933, 2132,
    #      1287, 590, 1190, 1290, 1671, 2908, 1083, 1381, 180, 2064, 864, 697, 998, 1105, 710, 2914, 1708, 1708,
    #      1626, 2034, 1733, 1361, 1514, 1809, 2529, 2010, 1313, 1710, 713, 1128, 713, 1011, 2417, 2405, 1302, 2622,
    #      1207, 1603, 2204, 705, 1560, 1713, 1273, 4968, 2673, 1479, 297, 1483, 285, 581, 2297, 7584, 5087, 2574,
    #      1785, 2182, 274, 561, 963, 1263, 2479, 1471, 1070, 1967, 1465, 2530, 150, 2838, 845, 1141, 1546, 1805,
    #      1409, 1012, 1418, 2625, 793, 1223, 705, 1005, 1405]
    y = [86971, 86007, 85585, 85862, 86251, 85348, 86444, 86812, 85895, 85731, 86931, 107204, 105840, 104457,
         102491, 101925, 99457, 97793, 96339, 94962, 92998, 114488, 134451, 133081, 131111, 130471, 129383, 127222,
         125461, 124096, 122436, 120770, 119400, 118440, 138034, 136355, 135296, 133292, 131827, 130669, 128508,
         126853, 126974, 124318, 122949, 121287, 120426, 118167, 116504, 114834, 113574, 111750, 110091, 108429,
         106978, 105818, 103951, 102694, 100633, 99170, 97507, 95972, 94182, 92822, 91169, 89504, 88035, 88558,
         86782, 87920, 87150, 88279, 87415, 87812, 86854, 88079, 88031, 88349, 87485, 87122, 87058, 87993, 87322,
         87113, 87547, 88777, 87117, 87549, 87381, 86520, 87443, 88674, 88007, 87845, 87475, 87107, 87738, 86845,
         88101, 87212]

    # print("x len: " + str(len(x)))
    # print("y len: " + str(len(y)))
    # mp.scatter(np.array(x), np.array(y))
    # mp.show()
    # X_mean = np.mean(x)
    # Y_mean = np.mean(y)
    # num = 0
    # den = 0
    # for i in range(len(x)):
    #     num += (x[i] - X_mean) * (y[i] - Y_mean)
    #     den += (x[i] - X_mean) ** 2
    # m = num / den
    # c = Y_mean - m * X_mean
    # print(m, c)
    # Y_pred = m * np.array(x) + c
    # mp.scatter(x, y)
    # mp.scatter(x, Y_pred, color='red')
    # mp.show()
    # w = np.std(np.array(y))
    # print(w)
    # mod = stm.WLS(np.array(y), np.array(x), weights=1./w**2)
    # res = mod.fit().params
    print(len(y))
    decreasing_trend_filter(y)
