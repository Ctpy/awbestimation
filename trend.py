import numpy as np
import math
import globals
import matplotlib.pyplot as mp


def calculate_trend(timestamps, packet_loss, train_length):
    np.set_printoptions(suppress=True)
    # Evaluate timestamps

    # Decreasing Trend due to buffering DT filter to filter out large bursts
    pct_metric_flag, pdt_metric_flag = decreasing_trend_filter(timestamps, packet_loss, train_length)

    if pct_metric_flag == "INCR" and pdt_metric_flag == "INCR":
        return "INCR", pct_metric_flag, pdt_metric_flag

    # Robust Regression filter to filter out small bursts by using Iteratively Re-weighted Least Square method (IRLS)

    pct_metric_flag, pdt_metric_flag = robust_regression_filter(timestamps, packet_loss, train_length)
    # Evaluate trend
    if pct_metric_flag == "INCR" and pdt_metric_flag == "INCR":
        return "INCR", pct_metric_flag, pdt_metric_flag
    elif pct_metric_flag == "INCR" and pdt_metric_flag == "UNCL":
        return "INCR", pct_metric_flag, pdt_metric_flag
    elif pct_metric_flag == "UNCL" and pdt_metric_flag == "INCR":
        return "INCR", pct_metric_flag, pdt_metric_flag
    else:
        return "NOCHANGE", pct_metric_flag, pdt_metric_flag


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


def decreasing_trend_filter(timestamps, packet_loss, train_length):
    pdt_flag = ""
    pct_flag = ""
    # search for burst
    mean = np.mean(timestamps)
    standard_derivation = compute_standard_derivation(timestamps)
    burst_packet_index_list = []
    for i in range(len(timestamps) - globals.DT_CONSECUTIVE):
        if mean + standard_derivation < timestamps[i]:
            burst_packet_index_list.append(i)

    # search for consecutive packet sample with decreasing rtt
    decreasing_trend_index_list = []
    for i in range(len(burst_packet_index_list)):
        decreasing_trend = True
        last_packet_rtt = timestamps[burst_packet_index_list[i]]
        for j in range(globals.DT_CONSECUTIVE):
            if last_packet_rtt < timestamps[burst_packet_index_list[i] + j]:
                decreasing_trend = False
                break
        if decreasing_trend:
            decreasing_trend_index_list.append(burst_packet_index_list[i])

    # remove all bursts
    for i in range(len(decreasing_trend_index_list)):
        for j in range(globals.DT_CONSECUTIVE):
            del timestamps[
                decreasing_trend_index_list[i]:decreasing_trend_index_list[i] + globals.DT_CONSECUTIVE]

    # pct

    if pct_metric(timestamps, train_length - len(timestamps), len(timestamps) > globals.BOUNDARY_PCT * 1.1):
        pct_flag = "INCR"
    elif pct_metric(timestamps, train_length - len(timestamps), len(timestamps) > globals.BOUNDARY_PCT * 0.9):
        pct_flag = "UNCL"
    else:
        pct_flag = "NOCHANGE"
    # pdt

    if pdt_metric(timestamps, train_length - len(timestamps), len(timestamps) > globals.BOUNDARY_PDT * 1.1):
        pdt_flag = "INCR"
    elif pdt_metric(timestamps, train_length - len(timestamps), len(timestamps) > globals.BOUNDARY_PDT * 0.9):
        pdt_flag = "UNCL"
    else:
        pdt_flag = "NOCHANGE"

    # return trend

    return pct_flag, pdt_flag


def robust_regression_filter(timestamps, packet_loss, train_length):
    iter_limit = 10
    columns = 2
    x_matrix = np.zeros(shape=(train_length - packet_loss, columns))
    y_matrix = np.array(timestamps)
    residual = 0.0
    q_matrix = np.zeros(shape=(train_length - packet_loss, columns))
    r_matrix = np.zeros(shape=(columns, columns))
    inv_r = np.zeros(shape=(columns, columns))
    e_matrix = np.zeros(shape=(train_length - packet_loss, columns))
    weights = []
    adjacent_factor = np.zeros(shape=(train_length - packet_loss, 1))
    r_adjacent_factor = np.zeros(shape=(train_length - packet_loss, 1))
    j = 0
    for i in range(train_length):
        if timestamps[i] is None:
            continue
        else:
            x_matrix[i, 0] = 1.0
            x_matrix[i, 1] = i + 1
            j += 1
    print(x_matrix)
    # q_matrix, r_matrix = np.linalg.qr(x_matrix)
    r_matrix, q_matrix = qr_decomposition_modified(x_matrix)
    print(q_matrix)
    print(r_matrix)
    e_matrix = compute_q_matrix(x_matrix, np.linalg.inv(r_matrix))
    print(e_matrix)
    adjacent_factor = compute_adjacent_factor(e_matrix)
    print(adjacent_factor)
    print("STD")
    print(np.std(y_matrix))
    # print(compute_standard_derivation(y_matrix))
    m, c, sigma_a, sigma_b, chi_square = least_square_fit(y_matrix)
    print("m: {} | c: {} | sigma_a: {} | sigma_b: {} | chi_square: {}".format(m, c, sigma_a, sigma_b, chi_square))
    small_derivation = 10e-6 * compute_standard_derivation(y_matrix)
    if small_derivation == 0:
        small_derivation = 1
    m_re_weighted = 0
    c_re_weighted = 0
    while iter == 0 or m - m_re_weighted > 1.4901e-08 * max(abs(m),
                                                            abs(m_re_weighted)) or c - c_re_weighted > 1.4901e-08 * max(
            abs(c), c_re_weighted):
        if iter + 1 > iter_limit:
            break

        # compute residuals
        residual_vector = []
        for i in range(len(y_matrix)):
            residual_vector.append(adjacent_factor[i] * (y_matrix[i] - (c + x_matrix[i][1] * m)))

        new_scale_factor = compute_median_absolute_derivation_sigma(np.array(residual_vector))

        for i in range(len(y_matrix)):
            residual = residual_vector[i] / max(new_scale_factor, small_derivation)
            if abs(residual) < 1:
                weights.append((1 - residual ** 2) ** 2)
            else:
                weights.append(0)
        m_re_weighted = m
        c_re_weighted = c

        c, m = weighted_least_square_fit(y_matrix, weights)

    # Filter timestamps with weights
    filtered_timestamps = []
    for i in range(len(timestamps)):
        if weights[i] >= 0.7:
            filtered_timestamps.append(timestamps[i])

    if pct_metric(filtered_timestamps, train_length - len(filtered_timestamps), len(filtered_timestamps) > globals.BOUNDARY_PCT * 1.1):
        pct_flag = "INCR"
    elif pct_metric(filtered_timestamps, train_length - len(filtered_timestamps), len(filtered_timestamps) > globals.BOUNDARY_PCT * 0.9):
        pct_flag = "UNCL"
    else:
        pct_flag = "NOCHANGE"
    # pdt

    if pdt_metric(filtered_timestamps, train_length - len(filtered_timestamps), len(filtered_timestamps) > globals.BOUNDARY_PDT * 1.1):
        pdt_flag = "INCR"
    elif pdt_metric(filtered_timestamps, train_length - len(filtered_timestamps), len(filtered_timestamps) > globals.BOUNDARY_PDT * 0.9):
        pdt_flag = "UNCL"
    else:
        pdt_flag = "NOCHANGE"

    # return trend

    return pct_flag, pdt_flag


def least_square_fit(y_matrix):
    m = 0.0
    chi_square = 0.0
    sum_x = 0.0
    sum_y = 0.0
    error_2 = 0.0
    x_matrix = range(1, len(y_matrix) + 1)
    for i in range(len(y_matrix)):
        sum_x += x_matrix[i]
        sum_y += y_matrix[i]
    x_average = sum_x / len(y_matrix)

    for i in range(len(y_matrix)):
        error = x_matrix[i] - x_average
        error_2 += error ** 2
        m += error * y_matrix[i]

    m /= error_2
    c = (sum_y - sum_x * m) / len(y_matrix)
    sigma_a = math.sqrt((1.0 + sum_x ** 2 / (len(y_matrix) * error_2)) / len(y_matrix))
    sigma_b = math.sqrt(1.0 / error_2)

    for i in range(len(y_matrix)):
        chi_square += (y_matrix[i] - c - m * x_matrix[i]) ** 2
    sig_refactor = math.sqrt(chi_square / (len(y_matrix) - 2))
    sigma_a *= sig_refactor
    sigma_b *= sig_refactor
    return m, c, sigma_a, sigma_b, chi_square


def weighted_least_square_fit(y_matrix, weights):
    x_matrix = range(1, len(y_matrix) + 1)
    sum_x = 0.0
    sum_y = 0.0
    sum_x_y = 0.0
    sum_x_x = 0.0
    sum_weights = 0.0
    for i in range(len(y_matrix)):
        sum_x += x_matrix[i] * weights[i]
        sum_y += y[i] * weights[i]
        sum_x_x += x_matrix[i] ** 2 * weights[i]
        sum_x_y += x_matrix[i] * y_matrix[i] * weights[i]
        sum_weights += weights

    factor = sum_weights * sum_x_x - sum_x ** 2
    c = (sum_x_x * sum_y - sum_x * sum_x_y) / factor
    m = (sum_weights * sum_x_y - sum_x * sum_y) / factor

    return c, m


def qr_decomposition_modified(matrix):
    rows, columns = matrix.shape
    r_matrix = np.zeros(shape=(columns, columns))
    q_matrix = np.zeros(shape=matrix.shape)
    a_matrix = np.zeros(shape=matrix.shape)
    b_matrix = np.zeros(shape=matrix.shape)

    for i in range(rows):
        for j in range(columns):
            b_matrix[i, j] = matrix[i, columns - 1 - j]

    r_matrix, q_matrix = gram_schmidt_modified(a_matrix, b_matrix, r_matrix, rows, columns)
    return -r_matrix, -q_matrix


def dot_product(u_matrix, v_matrix, k, j, rows):
    sum = 0.0
    for i in range(rows):
        sum += u_matrix[i, k] * v_matrix[i, j]
    return sum


def norm_l2(x_matrix, column, rows):
    sum = 0.0
    for i in range(rows):
        sum += (x_matrix[i, column] * x_matrix[i, column])
    return math.sqrt(sum)


def gram_schmidt_modified(q_matrix, x_matrix, r_matrix, rows, columns):
    tmp_matrix = x_matrix.copy()

    for i in range(columns):
        r_matrix[i, i] = norm_l2(tmp_matrix, i, rows)

        if r_matrix[i, i] == 0.0:
            break
        else:
            for j in range(rows):
                result = tmp_matrix[j, i] / r_matrix[i, i]
                print(result)
                q_matrix[j, i] = result

        for k in range(i + 1, columns):
            r_matrix[i, k] = dot_product(tmp_matrix, q_matrix, k, i, rows)
            for j in range(rows):
                tmp_matrix[j, k] -= r_matrix[i, k] * q_matrix[j, i]
    return r_matrix, q_matrix


def compute_q_matrix(x_matrix, r_matrix):
    rows, columns = x_matrix.shape
    e_matrix = np.zeros(shape=x_matrix.shape)
    for i in range(rows):
        for j in range(2):
            for k in range(2):
                e_matrix[i, j] += x_matrix[i, 1 - k] * r_matrix[k, j]
    return e_matrix


def compute_adjacent_factor(e_matrix):
    adjacent_factor = []
    rows, columns = e_matrix.shape
    for i in range(rows):
        factor = 0
        if (e_matrix[i, 0] * e_matrix[i, 0] + e_matrix[i, 1] * e_matrix[i, 1]) > 0.9999:
            factor = 1.0 / math.sqrt(1 - 0.9999)
        else:
            factor = 1.0 / math.sqrt(1 - (e_matrix[i, 0] * e_matrix[i, 0] + e_matrix[i, 1] * e_matrix[i, 1]))
        adjacent_factor.append(factor)
    return np.array(adjacent_factor)


def compute_median_absolute_derivation_sigma(data_vector):
    residual_data_vector = []
    for i in range(len(data_vector)):
        residual_data_vector.append(abs(data_vector[i]))
    residual_data_vector = np.sort(residual_data_vector)
    if len(data_vector) % 2 == 0:
        return residual_data_vector[(len(data_vector) - 1) / 2] / 0.675
    else:
        return (residual_data_vector[(len(data_vector) - 1) / 2] + residual_data_vector[
            (len(data_vector) + 1) / 2]) / 2.0 / 0.675


def compute_standard_derivation(data_set):
    mean = 0.0
    variance = 0.0
    for i in range(len(data_set)):
        mean += data_set[i]
    print("Mean Sum: " + str(mean))
    print("len: " + str(len(data_set)))
    mean /= (len(data_set))
    print("Mean: " + str(mean))
    for i in range(len(data_set)):
        variance += (data_set[i] - mean) ** 2
    variance /= (len(data_set))
    print("Variance: " + str(variance))
    return math.sqrt(variance)


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
    robust_regression_filter(y, 0, 100)
