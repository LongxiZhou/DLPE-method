import numpy as np
import os


def get_father_dict():
    return os.path.abspath(os.path.join(os.getcwd(), '..'))


def get_current_dict():
    return os.getcwd()


def save_np_array(dict, file_name, np_array, compress=False):
    # if the dict not exist, we make the dict
    if not os.path.exists(dict):
        os.makedirs(dict)
    if not compress:
        np.save(dict + file_name, np_array)
    else:
        np.savez_compressed(dict + file_name, array=np_array)


def f1_sore_for_binary_mask(prediction, ground_truth):
    prediction = np.array(prediction > 0.5, 'float32')
    ground_truth = np.array(ground_truth > 0.5, 'float32')
    over_lap = np.sum(prediction * ground_truth)
    return 2 * over_lap / (np.sum(prediction) + np.sum(ground_truth))


def recall(prediction, ground_truth):
    prediction = np.array(prediction > 0.5, 'float32')
    ground_truth = np.array(ground_truth > 0.5, 'float32')
    over_lap = np.sum(prediction * ground_truth)
    total_positive = np.sum(ground_truth)
    return over_lap / total_positive

