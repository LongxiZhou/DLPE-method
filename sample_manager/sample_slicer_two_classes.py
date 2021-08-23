import numpy as np
import Tool_Functions.Functions as Functions
import os


def slice_one_sample(data, resolution, slice_index, direction, ground_truth=None, window=(-5, -2, 0, 2, 5)):
    # raw_array has shape like [512, 512, 512] for [x, y, z],
    # resolution is a tuple like (334/512, 334/512, 1) in our TMI paper
    # window refers to the physical distance in mm
    # slice_index is the offset of the window center
    # return a sample with shape like [512, 512, len(window) + 1], the last channel is the ground_truth
    # if no ground truth, return a sample with shape like [512, 512, len(window)], for prediction
    num_input_channel = len(window)
    shape = np.shape(data)
    sample = None
    length = None
    if direction == 'X':
        resolution = resolution[0]
        sample = np.zeros([shape[1], shape[2], num_input_channel + 1], 'float32')
        length = shape[0]
    if direction == 'Y':
        resolution = resolution[1]
        sample = np.zeros([shape[0], shape[2], num_input_channel + 1], 'float32')
        length = shape[1]
    if direction == 'Z':
        resolution = resolution[2]
        sample = np.zeros([shape[0], shape[1], num_input_channel + 1], 'float32')
        length = shape[2]
    assert sample is not None
    assert length is not None
    if ground_truth is not None:
        sample[:, :, num_input_channel] = ground_truth
    else:
        sample = sample[:, :, 0: num_input_channel]
    if length > slice_index >= 0:  # slice_index is the ground truth mask of the input data
        if direction == 'X':
            for index in range(num_input_channel):
                slice_id = int(window[index]/resolution) + slice_index
                if 0 <= slice_id < length:
                    sample[:, :, index] = data[slice_id, :, :]
        if direction == 'Y':
            for index in range(num_input_channel):
                slice_id = int(window[index]/resolution) + slice_index
                if 0 <= slice_id < length:
                    sample[:, :, index] = data[:, slice_id, :]
        if direction == 'Z':
            for index in range(num_input_channel):
                slice_id = int(window[index]/resolution) + slice_index
                if 0 <= slice_id < length:
                    sample[:, :, index] = data[:, :, slice_id]
    return sample


def slice_one_direction(raw_array, resolution, direction, window=(-5, -2, 0, 2, 5), neglect_negative=True, threshold=5):
    # raw_array has shape [512, 512, 512, 2] for [x, y, z, -], or [512, 512, 512] for [x, y, z]
    # window refers to the physical distance in mm
    # return a list of samples
    data = None
    mask = None
    shape = np.shape(raw_array)
    if len(shape) == 4:  # raw_array is a rescaled data with mask
        data = raw_array[:, :, :, 0]
        mask = raw_array[:, :, :, 1]
    if len(shape) == 3:  # raw_array is a rescaled data for prediction or unsupervised learning
        data = raw_array
    assert data is not None
    sample_list = []
    if direction == 'X':
        length = shape[0]
        for index in range(length):
            if len(shape) == 3:
                if neglect_negative:
                    if np.sum(data[index, :, :]) <= threshold:
                        continue
                sample_list.append(slice_one_sample(data, resolution, index, direction, window=window))
            if len(shape) == 4:
                ground_truth = mask[index, :, :]
                if neglect_negative:
                    if np.sum(ground_truth) <= threshold:
                        continue
                sample_list.append(slice_one_sample(data, resolution, index, direction, ground_truth, window=window))
    if direction == 'Y':
        length = shape[1]
        for index in range(length):
            if len(shape) == 3:
                if neglect_negative:
                    if np.sum(data[:, index, :]) <= threshold:
                        continue
                sample_list.append(slice_one_sample(data, resolution, index, direction, window=window))
            if len(shape) == 4:
                ground_truth = mask[:, index, :]
                if neglect_negative:
                    if np.sum(ground_truth) <= threshold:
                        continue
                sample_list.append(slice_one_sample(data, resolution, index, direction, ground_truth, window=window))
    if direction == 'Z':
        length = shape[2]
        for index in range(length):
            if len(shape) == 3:
                if neglect_negative:
                    if np.sum(data[:, :, index]) <= threshold:
                        continue
                sample_list.append(slice_one_sample(data, resolution, index, direction, window=window))
            if len(shape) == 4:
                ground_truth = mask[:, :, index]
                if neglect_negative:
                    if np.sum(ground_truth) <= threshold:
                        continue
                sample_list.append(slice_one_sample(data, resolution, index, direction, ground_truth, window=window))
    return sample_list


def prepare_training_set(dict_for_arrays_raw, save_dict, resolution=(1, 1, 1), window=(-1, 0, 1), threshold=0):
    arrays_raw_name_list = os.listdir(dict_for_arrays_raw)
    num_scans = len(arrays_raw_name_list)
    print('there are', num_scans, 'of scans')

    scan_left = num_scans
    for arrays_raw_name in arrays_raw_name_list:
        print(scan_left, 'number of scans waiting to slicing')
        arrays_raw = np.load(dict_for_arrays_raw + arrays_raw_name)
        sample_list_x = slice_one_direction(arrays_raw, resolution, 'X', window, threshold=threshold)
        sample_list_y = slice_one_direction(arrays_raw, resolution, 'Y', window, threshold=threshold)
        sample_list_z = slice_one_direction(arrays_raw, resolution, 'Z', window, threshold=threshold)
        print('scan', arrays_raw_name, 'has:')
        print((len(sample_list_x), len(sample_list_y), len(sample_list_z)), 'samples from (X, Y, Z)')
        save_count = 0
        for sample in sample_list_x:
            save_count += 1
            Functions.save_np_array(save_dict + 'X/', 'X_' + str(save_count) + '_' + arrays_raw_name, sample)
        save_count = 0
        for sample in sample_list_y:
            save_count += 1
            Functions.save_np_array(save_dict + 'Y/', 'Y_' + str(save_count) + '_' + arrays_raw_name, sample)
        save_count = 0
        for sample in sample_list_z:
            save_count += 1
            Functions.save_np_array(save_dict + 'Z/', 'Z_' + str(save_count) + '_' + arrays_raw_name, sample)
        scan_left -= 1


if __name__ == '__main__':
    #prepare_training_set('/ibex/scratch/projects/c2052/air_tube_seg/arrays_raw/', '/ibex/scratch/projects/c2052/air_tube_seg/training_samples/')
    files = os.listdir('/home/zhoul0a/Desktop/air_tube_seg/balance_weight_array/X/')
    for f in files:
        sample = np.load(
            '/home/zhoul0a/Desktop/air_tube_seg/balance_weight_array/X/' + f)
        print(np.shape(sample))
        Functions.image_show(sample[:, :, 0])
        Functions.image_show(sample[:, :, 1])
        # Functions.image_show(sample[:, :, 1] + sample[:, :, 3])



