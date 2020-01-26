import numpy as np
import math
import globals
import matplotlib.pyplot as mp
import utility

#import statsmodels.api as sm
mp.switch_backend('agg')

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


def decreasing_trend_filter(timestamps_tuple, verbose):
    sent_time, timestamps = zip(*timestamps_tuple)
    # search for burst
    sent_time = list(sent_time)
    timestamps = list(timestamps)
    mean = np.mean(timestamps)
    standard_derivation = np.std(timestamps)
    burst_packet_index_list = [0]
    utility.print_verbose("Mean: " + str(mean), verbose)
    utility.print_verbose("Standard Derivation: " + str(standard_derivation), verbose)
    # search for consecutive packet sample with decreasing rtt
    decreasing_trend_index_list = []
    for i in range(len(timestamps)):
        consecutive = 0
        tmp = []
        last_packet_index = i
        for j in range(i + 1, len(timestamps)):
            if timestamps[last_packet_index] > timestamps[j]:
                consecutive += 1
                tmp.append(last_packet_index)
                last_packet_index = j
            else:
                tmp.append(last_packet_index)
                break
        if len(tmp) >= globals.DT_CONSECUTIVE:
            decreasing_trend_index_list.extend(tmp)
            i = last_packet_index

    timestamps = np.array(timestamps)
    timestamps = np.delete(timestamps, decreasing_trend_index_list)
    sent_time = np.array(sent_time)
    sent_time = np.delete(sent_time, decreasing_trend_index_list)
    timestamps = zip(tuple(sent_time), tuple(timestamps))
    timestamps.sort(key=lambda tup: tup[0])
    return timestamps, decreasing_trend_index_list, standard_derivation


def robust_regression_filter(timestamps, slope, constant):
    iter_limit = 10
    columns = 2
    m = slope
    c = constant
    for i in range(iter_limit):
        weight_vector = []
        for times in timestamps:
            error = math.fabs((m * times[0] + c) - times[1])
            weight_vector.append(1. / error ** 2)
        x_vector, Y = zip(*timestamps)
        X = sm.add_constant(x_vector)
        wls_model = sm.WLS(Y, X, weights=weight_vector)
        results = wls_model.fit()
        print("Slope: " + str(results.params))
        new_c = results.params[0]
        new_m = results.params[1]
        if i > 0:
            converge = False
            for i in range(len(weight_vector)):
                if math.fabs(new_c - c) < 0.0001 and math.fabs(new_m - m) < 0.0001:
                    converge = True
                else:
                    converge = False
                    break
            if converge:
                break
        c = results.params[0]
        m = results.params[1]
    return c, m


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # x = list(map(float,
    #              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    #               28,
    #               29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
    #               54,
    #               55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    #               80,
    #               81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]))
    # y = [-1, 1856, 604, 1371, 1724, 0, 1873, 1456, 863, 537, 2418, 1215, 1012, 606, 1502, 1195, 1497, 2933, 2132,
    #      1287, 590, 1190, 1290, 1671, 2908, 1083, 1381, 180, 2064, 864, 697, 998, 1105, 710, 2914, 1708, 1708,
    #      1626, 2034, 1733, 1361, 1514, 1809, 2529, 2010, 1313, 1710, 713, 1128, 713, 1011, 2417, 2405, 1302, 2622,
    #      1207, 1603, 2204, 705, 1560, 1713, 1273, 4968, 2673, 1479, 297, 1483, 285, 581, 2297, 7584, 5087, 2574,
    #      1785, 2182, 274, 561, 963, 1263, 2479, 1471, 1070, 1967, 1465, 2530, 150, 2838, 845, 1141, 1546, 1805,
    #      1409, 1012, 1418, 2625, 793, 1223, 705, 1005, 1405]
    # y = [86971, 86007, 85585, 85862, 86251, 85348, 86444, 86812, 85895, 85731, 86931, 107204, 105840, 104457,
    #      102491, 101925, 99457, 97793, 96339, 94962, 92998, 114488, 134451, 133081, 131111, 130471, 129383, 127222,
    #      125461, 124096, 122436, 120770, 119400, 118440, 138034, 136355, 135296, 133292, 131827, 130669, 128508,
    #      126853, 126974, 124318, 122949, 121287, 120426, 118167, 116504, 114834, 113574, 111750, 110091, 108429,
    #      106978, 105818, 103951, 102694, 100633, 99170, 97507, 95972, 94182, 92822, 91169, 89504, 88035, 88558,
    #      86782, 87920, 87150, 88279, 87415, 87812, 86854, 88079, 88031, 88349, 87485, 87122, 87058, 87993, 87322,
    #      87113, 87547, 88777, 87117, 87549, 87381, 86520, 87443, 88674, 88007, 87845, 87475, 87107, 87738, 86845,
    #      88101, 87212]

    x = np.arange(0, 1, 0.005)
    y = np.array(
        [0.3, 0.5, 0.45, 0.40, 0.35, 0.37, 0.39, 0.41, 0.7, 0.67, 0.65, 0.63, 0.60, 0.59, 0.58, 0.57, 0.55, 0.53, 0.52,
         0.4, 0.45, 0.47, 0.50, 0.8, 0.51, 0.54, 0.55, 0.59, 0.9, 0.6, 0.62, 0.65, 0.66, 0.67,
         0.78, 0.76, 0.75, 0.72, 0.6, 0.5, 0.4, 0.9, 0.88, 0.87, 0.85, 0.80, 0.78, 0.77, 0.74, 0.70, 0.66, 0.65, 1.00,
         0.98, 0.96, 0.94, 0.67, 0.3, 0.5, 0.45, 0.40, 0.35, 0.37, 0.39, 0.41, 0.7, 0.67, 0.65, 0.63, 0.60, 0.59, 0.58,
         0.57, 0.55, 0.53, 0.52, 0.4, 0.45, 0.47, 0.50, 0.8, 0.51, 0.54, 0.55, 0.59, 0.9, 0.6, 0.62, 0.65, 0.66, 0.67,
         0.78, 0.76, 0.75, 0.72, 0.6, 0.5, 0.4, 0.9, 0.88, 0.87, 0.85, 0.80, 0.78, 0.77, 0.74, 0.70, 0.66, 0.65, 1.00,
         0.98, 0.96, 0.94, 0.67, 0.3, 0.5, 0.45, 0.40, 0.35, 0.37, 0.39, 0.41, 0.7, 0.67, 0.65, 0.63, 0.60, 0.59, 0.58,
         0.57, 0.55, 0.53, 0.52, 0.4, 0.45, 0.47, 0.50, 0.8, 0.51, 0.54, 0.55, 0.59, 0.9, 0.6, 0.62, 0.65, 0.66, 0.67,
         0.78, 0.76, 0.75, 0.72, 0.6, 0.5, 0.4, 0.9, 0.88, 0.87, 0.85, 0.80, 0.78, 0.77, 0.74, 0.70, 0.66, 0.65, 1.00,
         0.98, 0.96, 0.94, 0.87, 0.85, 0.80, 0.78, 0.75, 0.73, 0.67, 0.60, 0.5, 1.2, 1.11, 1.05, 1.01, 0.91, 0.85, 0.76,
         0.7, 0.66, 0.63, 0.59, 0.56, 0.55, 0.52, 0.50, 0.49, 0.45, 0.40, 0.37, 0.20, 0.20])
    y_decreasing_trend = np.array([0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.7, 0.39, 0.38, 0.36, 0.34, 0.32, 0.33,
                                   0.35, 0.37, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44, 0.43, 0.42, 0.41, 0.4, 0.42, 0.44,
                                   0.45, 0.46, 0.9, 0.78, 0.66, 0.65, 0.63, 0.6, 0.59, 0.58, 0.55, 0.54, 0.51, 0.5, 0.52,
                                   0.53, 0.55, 0.56, 1.0, 0.79, 0.77, 0.75, 0.73, 0.70, 0.69, 0.68, 0.66, 0.65, 0.64, 0.63,
                                   0.60, 0.57, 0.6, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 1.2, 1.08, 0.96, 0.85,
                                   0.84, 0.83, 0.82, 0.81, 0.8, 0.78, 0.74, 0.72, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.78,
                                   0.8, 1.2, 1.15, 1.10, 1.08, 1.05, 1.03, 1.01, 1.00, 0.98, 0.94, 0.92, 0.90, 0.88, 0.87,
                                   0.86, 0.85, 0.87, 0.9, 0.92, 0.94, 0.95, 0.97, 1.00, 1.5, 1.46, 1.34, 1.33, 1.32, 1.3,
                                   1.29, 1.27, 1.26, 1.25, 1.24, 1.22, 1.20, 1.18, 1.17, 1.16, 1.15, 1.14, 1.11, 1.1, 1.05,
                                   1.03, 1.06, 1.09, 1.11, 1.15, 1.17, 1.19, 1.5, 1.48, 1.46, 1.44, 1.40, 1.37, 1.32, 1.29,
                                   1.25, 1.27, 1.31, 1.32, 1.35, 1.38, 1.39, 1.8, 1.68, 1.55, 1.52, 1.5, 1.47, 1.45, 1.43,
                                   1.4, 1.39, 1.37, 1.35, 1.33, 1.39, 1.45, 1.49, 1.52, 1.55, 1.59, 2.1, 1.96, 1.82, 1.67,
                                   1.65, 1.6, 1.62, 1.69, 1.99, 1.96, 1.95, 1.93, 1.8, 1.82, 1.84, 2.4, 2.25, 2.1, 1.98,
                                   1.85, 1.83, 1.82, 1.8, 1.78])
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
    timestamps = zip(x, y_decreasing_trend)
    print(timestamps)
    new_timestamps, filtered, std = decreasing_trend_filter(timestamps, True)
    # Plot here
    mp.figure(figsize=(12,6))
    mp.plot(x, y_decreasing_trend, 'green', label="unfiltered", marker='x')
    mp.plot(*zip(*new_timestamps), color='blue', label="filtered", marker='x')
    mp.tick_params(axis='x', which='major')
    mp.savefig('test_dt.svg', format='svg')
    mp.show()
    mp.clf()
