import numpy as np
import matplotlib.pyplot as plt
import math
import visualization.visiualize_2d.data_point_visualization as visualization
import random


def poly_fit_and_show(x, y, order, save_path=None, show=True):

    z1 = np.polyfit(x, y, order)
    p1 = np.poly1d(z1)
    print(p1)  # 在屏幕上打印拟合多项式
    if show:
        yvals = p1(x)
        plt.plot(x, y, '*', label='original values')
        plt.plot(x, yvals, 'r', label='polyfit values')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.legend(loc=4)  # 指定legend的位置
        plt.title('polyfitting')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
    return z1


def linear_fit(x, y, show=True):
    n = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for i in range(0, int(n)):
        sx += x[i]
        sy += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/n - sxy)/(sx*sx/n - sxx)
    b = (sy - a*sx)/n
    r = (sy*sx/n-sxy)/math.sqrt((sxx-sx*sx/n)*(syy-sy*sy/n))
    if show:
        print("the fitting result is: y = %10.5f x + %10.5f , r = %10.5f" % (a, b, r))
    return a, b, r


def linear_regression(x, y, normalize_std_x=True, normalize_std_y=False):
    from scipy import stats
    if normalize_std_x:
        x_value = x / np.std(x)
    else:
        x_value = np.array(x)
    if normalize_std_y:
        y_value = y / np.std(y)
    else:
        y_value = np.array(y)
    slope, intercept, r_value, p_value, std_err_slope = stats.linregress(x_value, y_value)
    return slope, intercept, r_value, p_value, std_err_slope


def log_log_linear_fit(scale_list, frequency, cache=10, neglect_ratio=0.001, default=False, show=True):
    # scale_list: must in [1, 2, 3, 4, ...]
    # frequency is a list [frequency at 1, frequency at 2, ...]
    # neglect_ratio means we overlook a ratio of instance with largest scales
    if default:
        visualization.show_data_points(np.log(scale_list), np.log(frequency))
        linear_fit(np.log(scale_list), np.log(frequency + 1))
        return None
    length = len(scale_list)
    assert length == len(frequency)
    total_instance = np.sum(frequency)
    if show:
        print("the length of the scale is:", length)
        print("total instance is", total_instance)
    num_neglect = int(total_instance * neglect_ratio)
    biggest_scale = length - 1
    while num_neglect > 0:
        num_neglect -= frequency[biggest_scale]
        biggest_scale -= 1
    biggest_value = biggest_scale

    step = round(biggest_value / cache)

    if show:
        print("the biggest_scale being considered is:", biggest_value)
        print("each cache is:", step)

    frequent_list = np.zeros([cache + 2, ], 'float32')
    index = 0
    for scale in range(0, biggest_value, step):
        for loc in range(scale, scale + step):
            if loc >= biggest_value:
                continue
            if index >= cache:
                break
            frequent_list[index] += frequency[loc]
        index += 1

    for index in range(0, cache):
        if frequent_list[index] == 0:
            frequent_list[index] = 1
        frequent_list[index] = np.log(frequent_list[index])
    value_list = []
    for index in range(0, cache + 1):
        value_list.append(math.log(index * step + int(step/2)))
    if show:
        print(np.shape(value_list), np.shape(frequent_list), step)
        visualization.show_data_points(value_list[0: cache], frequent_list[0: cache])
    return linear_fit(value_list[0: cache], frequent_list[0: cache])


def scale_free_check(scale_list, frequency, cache=10, show=True):
    # scale_list is a ordered list recording the measurements, like area, degree, etc
    # frequency is a list recording the frequency or probability of each scale
    # cache is the number of output points
    scale_list = np.array(scale_list)
    frequency = np.array(frequency)
    length = len(scale_list)
    assert len(scale_list) == len(frequency)
    if show:
        print("the length of the list is", length)
    step = round(length/cache)

    def get_center(sub_list_scale, sub_list_frequency):
        return sum(sub_list_scale * sub_list_frequency)/sum(sub_list_frequency)

    center_list = []
    total_frequency_list = []
    for loc in range(0, length, step):
        if loc + step >= length:
            end = length
        else:
            end = loc + step
        list_cache_scale = scale_list[loc: end]
        list_cache_frequency = frequency[loc: end]
        center_list.append(get_center(list_cache_scale, list_cache_frequency))
        total_frequency = np.sum(list_cache_frequency)
        if total_frequency == 0:
            print("detect 0 frequency, replace with 1")
            total_frequency = 1
        total_frequency_list.append(total_frequency)
    if show:
        visualization.show_data_points(np.log(center_list), np.log(total_frequency_list))
    return linear_fit(np.log(center_list), np.log(total_frequency_list))


def chi2_contigency_test(list_a, list_b, a_level=3, b_level=3):
    """
    we have two different variables: list_a, list_b. test whether they are independent.
    :param list_a: variable a
    :param list_b: variable b
    :param a_level: level of variable a
    :param b_level: level of variable b
    :return: p value
    """
    from scipy.stats import chi2_contingency

    length_a = len(list_a)
    length_b = len(list_b)
    assert length_a == length_b
    interval_a = round(length_a / a_level)
    interval_b = round(length_b / b_level)

    p_value_log = 0

    tested_num = 0
    potential_p = []

    for j in range(1000):
        list_a_new = list(list_a)
        list_b_new = list(list_b)
        for i in range(length_a):  # add a small noise to make every observation distinguishable
            list_a_new[i] = list_a[i] + random.random() / 10000000
            list_b_new[i] = list_b[i] + random.random() / 10000000
        sorted_a = list(list_a_new)
        sorted_a.sort()
        sorted_b = list(list_b_new)
        sorted_b.sort()

        contigency_array = np.zeros([a_level, b_level], 'int32')
        for i in range(length_a):  # patient i
            value_a = list_a_new[i]
            value_b = list_b_new[i]
            loc_a = sorted_a.index(value_a)
            loc_b = sorted_b.index(value_b)
            contigency_array[min(int(loc_a / interval_a), a_level - 1), min(int(loc_b / interval_b), b_level - 1)] += 1
        current_p_log = math.log(chi2_contingency(contigency_array)[1])
        p_value_log = p_value_log + current_p_log
        tested_num += 1
        if current_p_log not in potential_p:
            potential_p.append(p_value_log)
        if tested_num % 100 == 9:
            if np.std(potential_p) / tested_num < 0.01:
                # print("converged at", tested_num - 8)
                break

    p_value_log = p_value_log / tested_num

    return math.exp(p_value_log)


def dependency_test_permutation(list_a, list_b, permutation_number=1000):
    """
    test whether variable x and variable y are independent
    use the abs(positive_correlation_pairs - negative_correlation_pairs) as indicator
    :param list_a: observations for variable x
    :param list_b: observations for variable y
    :param permutation_number:
    :return: probability of dependent
    """
    length = len(list_a)
    assert length == len(list_b)
    list_a = list(list_a)

    def pair_wise_check(list_c, list_d):
        positive_correlation_pairs = 0
        negative_correlation_pairs = 0
        for i in range(length):
            for j in range(i):
                x_i = list_c[i]
                x_j = list_c[j]
                y_i = list_d[i]
                y_j = list_d[j]
                if (x_i - x_j) * (y_i - y_j) > 0:
                    positive_correlation_pairs += 1
                if (x_i - x_j) * (y_i - y_j) < 0:
                    negative_correlation_pairs += 1
        total_compare = positive_correlation_pairs + negative_correlation_pairs
        more_extreme = total_compare - max([positive_correlation_pairs, negative_correlation_pairs]) - 1
        return more_extreme

    extreme_level_original = pair_wise_check(list_a, list_b)
    num_randoms_more_extreme = 0
    for t in range(permutation_number):
        random.shuffle(list_a)
        if pair_wise_check(list_a, list_b) < extreme_level_original:
            num_randoms_more_extreme += 1

    return num_randoms_more_extreme / permutation_number
