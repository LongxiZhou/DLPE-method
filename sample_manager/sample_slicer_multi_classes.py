import numpy as np
import Tool_Functions.Functions as Functions
import os

'''
these functions prepare 2D training samples for the semantic segmentation tasks.

Terminology: 
rescaled_array: the spatial and signal normalized array of images and the masks, during training, it is in shape 
[512, 512, 512, data_channel + enhanced_channel + semantic_channel]. e.g. CT has only one data channel while DCE-MRI may 
have multiple; e.g. there are 5 types of lesions and two types of normal tissue, thus semantic_channel is 7. e.g. during 
further fine-tuning of our three-way model, we may input the intermediate results of other directions, which increases 
the enhanced_channel.
During training, it is in shape [512, 512, 512, data_channel * len(window) + enhanced_channel], as you do not know the 
ground truth. Note, data_channel, enhanced_channel are int

resolution: the spatial normalization will cast the data into a standard resolution. Like in our TMI paper, the standard
resolution is (334/512, 334/512, 1), this means, each voxel indicates a volume of 334/512x334/512x1 cubic millimeter.

window: we may use adjacent slices to help the segmentation of the central slice. The window means the slicing window 
when making the samples. It is a tuple, indicating the absolute distance (in millimeters) e.g. (-5, -2, 0, 2, 5), means
we use 5 slices to predict the central slice (distance = 0). Note, the function used int(window/resolution_on_this_axes)
and the window is for 

direction: we slice the array from three directions, the legal inputs are 'X', 'Y' or 'Z'.

slice_index: it is the id of the image that the model aimed to segment. Your model input a sample, and output a 
segmentation map. Which image does the output segmentation map correspond to? It is the image with id of "slice_index".

What the sample looks like?
the sample shape and combinations are determined by slice_one_sample(), which is flexible. Usually, the sample has
shape: [512, 512, data_channel * len(window) + enhanced_channel + semantic_channel].
if the model input is [Batch_Num, C, H, W], then [C, H, W] is [data_channel * len(window) + enhanced_channel, 512, 512]
the semantic_channel is the ground truth, which is used during training.  
'''


def slice_one_sample(rescaled_array, resolution, slice_index, direction, data_channel, enhanced_channel, window):
    # we will call this function both in training and testing.
    # the rescaled_array has shape: [512, 512, 512, data_channel + enhanced_channel + semantic_channel], in 'float32'
    # if semantic_channel is 0, which indicates we are during testing.
    # return a sample with shape like [512, 512, len(window) * data_channel + enhanced_channel + semantic_channel]

    shape = np.shape(rescaled_array)
    assert len(shape) == 4
    window_slice_num = len(window)
    num_input_channel = window_slice_num * data_channel + enhanced_channel
    semantic_channel = shape[3] - data_channel - enhanced_channel

    sample = None
    length = None
    ground_truth = None
    if direction == 'X':
        resolution = resolution[0]
        sample = np.zeros([shape[1], shape[2], num_input_channel + semantic_channel], 'float32')
        length = shape[0]
        if semantic_channel > 0:  # semantic_channel contains the ground truth
            ground_truth = rescaled_array[slice_index, :, :, data_channel + enhanced_channel::]
    if direction == 'Y':
        resolution = resolution[1]
        sample = np.zeros([shape[0], shape[2], num_input_channel + semantic_channel], 'float32')
        length = shape[1]
        if semantic_channel > 0:  # semantic_channel contains the ground truth
            ground_truth = rescaled_array[:, slice_index, :, data_channel + enhanced_channel::]
    if direction == 'Z':
        resolution = resolution[2]
        sample = np.zeros([shape[0], shape[1], num_input_channel + semantic_channel], 'float32')
        length = shape[2]
        if semantic_channel > 0:  # semantic_channel contains the ground truth
            ground_truth = rescaled_array[:, :, slice_index, data_channel + enhanced_channel::]
    assert sample is not None
    assert length is not None  # this action avoid illegal input of the direction.

    if ground_truth is not None:  # this means the rescaled array contains ground truth
        sample[:, :, num_input_channel::] = ground_truth

    if length > slice_index >= 0:  # slice_index is the id of the image that the model aimed to segment.
        if direction == 'X':
            for index in range(window_slice_num):
                slice_id = int(window[index] / resolution) + slice_index
                if 0 <= slice_id < length:
                    sample[:, :, index * data_channel: (index + 1) * data_channel] = \
                        rescaled_array[slice_id, :, :, 0: data_channel]  # for data channels
            sample[:, :, window_slice_num * data_channel: num_input_channel] = \
                rescaled_array[slice_index, :, :, data_channel: data_channel + enhanced_channel]  # for enhance channels

        if direction == 'Y':
            for index in range(window_slice_num):
                slice_id = int(window[index] / resolution) + slice_index
                if 0 <= slice_id < length:
                    sample[:, :, index * data_channel: (index + 1) * data_channel] = \
                        rescaled_array[:, slice_id, :, 0: data_channel]  # for data channels
            sample[:, :, window_slice_num * data_channel: num_input_channel] = \
                rescaled_array[:, slice_index, :, data_channel: data_channel + enhanced_channel]  # for enhance channels

        if direction == 'Z':
            for index in range(window_slice_num):
                slice_id = int(window[index] / resolution) + slice_index
                if 0 <= slice_id < length:
                    sample[:, :, index * data_channel: (index + 1) * data_channel] = \
                        rescaled_array[:, :, slice_id, 0: data_channel]  # for data channels
            sample[:, :, window_slice_num * data_channel: num_input_channel] = \
                rescaled_array[:, :, slice_index, data_channel: data_channel + enhanced_channel]  # for enhance channels
    else:
        print('slice_index out of range')
        return None

    return sample


def slice_one_direction(rescaled_array, resolution, direction, data_channel, enhanced_channel, window,
                        neglect_negative=False, positive_semantic_channel=None, threshold=1, neglect_interval=None):
    # we will call this function both in training and testing.
    # the rescaled_array has shape: [512, 512, 512, data_channel + enhanced_channel + semantic_channel], in 'float32'
    # window refers to the physical distance in mm
    # during training, sometimes we want to neglect the sample if the segmentation map is uniform (like all pixel
    # corresponds to only one class), thus neglect_negative=True; during testing, we need to slice all slices, thus
    # neglect_negative=False
    # positive_semantic_channel: is a list of integers, indicting which semantic channels are considered as lesion.
    # e.g. the first semantic channel is parenchyma, second is nodule, third is infection, fourth is tumor, then,
    # positive_semantic_channel = (1, 2, 3)
    # "threshold" is the threshold to determine whether the segmentation map is uniform.
    # return a list of samples

    shape = np.shape(rescaled_array)
    window_len = len(window)
    if len(shape) == 3:
        assert shape == (512, 512, 512)
        rescaled_array = np.reshape(rescaled_array, [512, 512, 512, 1])
        shape = (512, 512, 512, 1)
    semantic_channel = shape[3] - data_channel - enhanced_channel
    assert semantic_channel >= 0
    if semantic_channel > 0:
        train_period = True
        assert positive_semantic_channel is not None
    else:
        train_period = False
        assert positive_semantic_channel is None
        assert neglect_negative is False

    sample_list = []  # the elements are the samples.

    length = None
    if direction == 'X':
        length = shape[0]
    if direction == 'Y':
        length = shape[1]
    if direction == 'Z':
        length = shape[2]
    assert length is not None

    for slice_index in range(length):
        sample = slice_one_sample(rescaled_array, resolution, slice_index, direction, data_channel,
                                  enhanced_channel, window)
        # the sample with shape: [512, 512, len(window) * data_channel + enhanced_channel + semantic_channel]
        if train_period and neglect_negative and neglect_interval is None:
            lesion_pixels = 0
            for lesion_channel in positive_semantic_channel:
                lesion_pixels += np.sum(sample[:, :, window_len * data_channel + enhanced_channel + lesion_channel])
            if lesion_pixels <= threshold:  # this means this sample only contain normal tissue
                continue
        if train_period and neglect_negative and neglect_interval is not None:
            if slice_index % neglect_interval == 0:
                sample_list.append(sample)
                continue
            lesion_pixels = 0
            for lesion_channel in positive_semantic_channel:
                lesion_pixels += np.sum(sample[:, :, window_len * data_channel + enhanced_channel + lesion_channel])
            if lesion_pixels <= threshold:  # this means this sample only contain normal tissue
                continue
        sample_list.append(sample)

    return sample_list


def prepare_training_set(dict_for_rescaled_array, save_dict, data_channel, enhanced_channel, positive_semantic_channel,
                         resolution=(1, 1, 1), window=(-1, 0, 1), threshold=0, neglect_interval=None):
    # param dict_for_rescaled_array: the directory where rescaled arrays are stored. can be .npz or .npy
    # param save_dict: the directory where the training samples will be stored
    # data_channel, enhanced_channel are both int, means their number
    # positive_semantic_channel is a list, indicating the lesion channel in semantic_channels, like (1, 2, 3).
    # positive_semantic_channel is for neglect all normal slices.
    # the name of training samples will be: direction + str(slice_index) + name(rescaled_array)
    # their is a file recording the reports. Each line is:
    # rescaled_array_name num_samples_from_X num_samples_from_Y num_samples_from_Z

    rescaled_array_name_list = os.listdir(dict_for_rescaled_array)
    num_scans = len(rescaled_array_name_list)
    print('there are total', num_scans, 'of scans')

    if not dict_for_rescaled_array[-1] == '/':
        dict_for_rescaled_array = dict_for_rescaled_array + '/'
    if not save_dict[-1] == '/':
        save_dict = save_dict + '/'
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    if os.path.exists(save_dict + 'report.txt'):
        report = open(save_dict + 'report.txt', 'r')
        processed_list = []  # the rescaled_array that already processed
        for line in report.readlines():
            processed_list.append(list(line.split(' '))[0])
        for processed_name in processed_list:
            if processed_name in rescaled_array_name_list:
                rescaled_array_name_list.remove(processed_name)
                num_scans -= 1
        print('there are', len(processed_list), 'scans have been processed, and', num_scans, 'left.')
    else:
        report = open(save_dict + 'report.txt', 'w+')
    report.close()

    report = open(save_dict + 'report.txt', 'a')
    report.write('scan id, sample_num_x, sample_num_y, sample_num_z\n')
    scan_left = num_scans
    for rescaled_array_name in rescaled_array_name_list:
        print(scan_left, 'number of scans waiting to slicing')
        if rescaled_array_name[-1] == 'y':  # this means is .npy file
            rescaled_array = np.load(dict_for_rescaled_array + rescaled_array_name)
        elif rescaled_array_name[-1] == 'z':  # this means is .npz file
            rescaled_array = np.load(dict_for_rescaled_array + rescaled_array_name)['array']
        else:
            print('illegal filename for:', rescaled_array_name)
            raise ValueError('rescaled_array must be .npy or .npz file')

        sample_list_x = slice_one_direction(rescaled_array, resolution, 'X', data_channel, enhanced_channel, window,
                                            True, positive_semantic_channel, threshold, neglect_interval)
        sample_list_y = slice_one_direction(rescaled_array, resolution, 'Y', data_channel, enhanced_channel, window,
                                            True, positive_semantic_channel, threshold, neglect_interval)
        sample_list_z = slice_one_direction(rescaled_array, resolution, 'Z', data_channel, enhanced_channel, window,
                                            True, positive_semantic_channel, threshold, neglect_interval)
        print('scan', rescaled_array_name, 'has:')
        sample_num_x = len(sample_list_x)
        sample_num_y = len(sample_list_y)
        sample_num_z = len(sample_list_z)
        print(sample_num_x, sample_num_y, sample_num_z, 'samples from (X, Y, Z)')
        save_count = 0

        for sample in sample_list_x:
            save_count += 1
            Functions.save_np_array(save_dict + 'X/', 'X_' + str(save_count) + '_' + rescaled_array_name, sample)
        save_count = 0
        for sample in sample_list_y:
            save_count += 1
            Functions.save_np_array(save_dict + 'Y/', 'Y_' + str(save_count) + '_' + rescaled_array_name, sample)
        save_count = 0
        for sample in sample_list_z:
            save_count += 1
            Functions.save_np_array(save_dict + 'Z/', 'Z_' + str(save_count) + '_' + rescaled_array_name, sample)
        scan_left -= 1
        report.write(rescaled_array_name + ' ' + str(sample_num_x) + ' ' + str(sample_num_y) + ' ' + str(sample_num_z))
        report.write('\n')
    report.close()


def prepare_training_set_v2(dict_for_rescaled_array, dict_for_rescaled_gt, gt_channel, save_dict, data_channel,
                            enhanced_channel, positive_semantic_channel, resolution=(1, 1, 1), window=(-1, 0, 1), threshold=0, neglect_interval=None):
    # param dict_for_rescaled_array: the directory where rescaled arrays are stored. can be .npz or .npy
    # param save_dict: the directory where the training samples will be stored
    # data_channel, enhanced_channel are both int, means their number
    # positive_semantic_channel is a list, indicating the lesion channel in semantic_channels, like (1, 2, 3).
    # positive_semantic_channel is for neglect all normal slices.
    # the name of training samples will be: direction + str(slice_index) + name(rescaled_array)
    # their is a file recording the reports. Each line is:
    # rescaled_array_name num_samples_from_X num_samples_from_Y num_samples_from_Z

    rescaled_array_name_list = os.listdir(dict_for_rescaled_array)
    num_scans = len(rescaled_array_name_list)
    print('there are total', num_scans, 'of scans')

    if not dict_for_rescaled_array[-1] == '/':
        dict_for_rescaled_array = dict_for_rescaled_array + '/'
    if not save_dict[-1] == '/':
        save_dict = save_dict + '/'
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    if os.path.exists(save_dict + 'report.txt'):
        report = open(save_dict + 'report.txt', 'r')
        processed_list = []  # the rescaled_array that already processed
        for line in report.readlines():
            processed_list.append(list(line.split(' '))[0])
        for processed_name in processed_list:
            if processed_name in rescaled_array_name_list:
                rescaled_array_name_list.remove(processed_name)
                num_scans -= 1
        print('there are', len(processed_list), 'scans have been processed, and', num_scans, 'left.')
    else:
        report = open(save_dict + 'report.txt', 'w+')
    report.close()

    report = open(save_dict + 'report.txt', 'a')
    report.write('scan id, sample_num_x, sample_num_y, sample_num_z\n')
    scan_left = num_scans
    for rescaled_array_name in rescaled_array_name_list:
        print(scan_left, 'number of scans waiting to slicing')
        if rescaled_array_name[-1] == 'y':  # this means is .npy file
            rescaled_array = np.load(dict_for_rescaled_array + rescaled_array_name)
        elif rescaled_array_name[-1] == 'z':  # this means is .npz file
            rescaled_array = np.load(dict_for_rescaled_array + rescaled_array_name)['array']
        else:
            print('illegal filename for:', rescaled_array_name)
            raise ValueError('rescaled_array must be .npy or .npz file')
        gt_array = np.load(dict_for_rescaled_gt + rescaled_array_name[:-4] + '.npz')['array'][:, :, :, gt_channel]
        assert np.shape(rescaled_array) == (512, 512, 512)
        new_array = np.zeros([512, 512, 512, 2], 'float32')
        new_array[:, :, :, 0] = rescaled_array
        new_array[:, :, :, 1] = gt_array

        rescaled_array = new_array

        sample_list_x = slice_one_direction(rescaled_array, resolution, 'X', data_channel, enhanced_channel, window,
                                            True, positive_semantic_channel, threshold, neglect_interval)
        sample_list_y = slice_one_direction(rescaled_array, resolution, 'Y', data_channel, enhanced_channel, window,
                                            True, positive_semantic_channel, threshold, neglect_interval)
        sample_list_z = slice_one_direction(rescaled_array, resolution, 'Z', data_channel, enhanced_channel, window,
                                            True, positive_semantic_channel, threshold, neglect_interval)
        print('scan', rescaled_array_name, 'has:')
        sample_num_x = len(sample_list_x)
        sample_num_y = len(sample_list_y)
        sample_num_z = len(sample_list_z)
        print(sample_num_x, sample_num_y, sample_num_z, 'samples from (X, Y, Z)')
        save_count = 0
        if rescaled_array_name[-1] == 'z':
            rescaled_array_name = rescaled_array_name[:-4] + '.npy'
        for sample in sample_list_x:
            save_count += 1
            Functions.save_np_array(save_dict + 'X/', 'X_' + str(save_count) + '_' + rescaled_array_name, sample)
        save_count = 0
        for sample in sample_list_y:
            save_count += 1
            Functions.save_np_array(save_dict + 'Y/', 'Y_' + str(save_count) + '_' + rescaled_array_name, sample)
        save_count = 0
        for sample in sample_list_z:
            save_count += 1
            Functions.save_np_array(save_dict + 'Z/', 'Z_' + str(save_count) + '_' + rescaled_array_name, sample)
        scan_left -= 1
        report.write(rescaled_array_name + ' ' + str(sample_num_x) + ' ' + str(sample_num_y) + ' ' + str(sample_num_z))
        report.write('\n')
    report.close()


def clear_label(original_dict, target_dict):
    fn_list = os.listdir(original_dict)
    for fn in fn_list:
        path = os.path.join(original_dict, fn)
        sample = np.load(path)
        channels = np.shape(sample)[-1]
        sample[:, :, channels - 1] = sample[:, :, channels - 1] * 0
        Functions.save_np_array(target_dict, fn, sample, False)


if __name__ == '__main__':
    exit()