"""

call get_connect_region_2d function for connectivity for 2D

calculate area and rim length: abstract_connected_regions

Evaluate the scale free property of the ground truth
core function:
abstract_data_set(directory of the training dataset, save directory, channel_of_mask, scale=5000)
The training data sample shaped [hight, width, channels], if the second channel is gt, then channels_of_mask = 1
scale is the maximum scale being considered, the function will save to frequency array, one focused on the area
distribution, the other focused on the rim_length distribution, like
frequency_array_area = [frequency_of_area=0, frequency_of_area=1, frequency_of_area=2, ..., frequence_of_area>5000]
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import multiprocessing as mp
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import Tool_Functions.Functions as Functions

np.set_printoptions(precision=10, suppress=True)
np.set_printoptions(threshold=np.inf)
epsilon = 0.001
adjacency = 'strict'  # one of ['loose', 'strict']
ibex = False
if not ibex:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # use two V100 GPU


class DimensionError(Exception):
    def __init__(self, array):
        self.shape = np.shape(array)
        self.dimension = len(self.shape)

    def __str__(self):
        print("invalid dimension of", self.dimension, ", array has shape", self.shape)


class SemanticError(Exception):
    def __init__(self, string_like):
        self.value = string_like

    def __str__(self):
        print("invalid semantic of", self.value)


def check_global_parameters():
    print("adjacency type is:", adjacency)
    assert isinstance(adjacency, type('loose'))
    if not adjacency == 'loose' and not adjacency == 'strict':
        raise SemanticError(adjacency)


class GetRimLoose(nn.Module):
    # if the adjacency is loosely defined, use this to get rim
    def __init__(self):
        super(GetRimLoose, self).__init__()
        super().__init__()
        kernel = [[[[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=1)
        return x


class GetRimStrict(nn.Module):
    # if the adjacency is strictly defined, use this to get rim
    def __init__(self):
        super(GetRimStrict, self).__init__()
        super().__init__()
        kernel = [[[[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=1)
        return x


if adjacency == 'loose':
    convolution_layer = GetRimLoose()
if adjacency == 'strict':
    convolution_layer = GetRimStrict()

convolution_layer = convolution_layer.cuda()

if torch.cuda.device_count() > 1:
    convolution_layer = nn.DataParallel(convolution_layer)


def func_parallel(func, list_inputs, leave_cpu_num=1):
    """
    :param func: func(list_inputs[i])
    :param list_inputs: each element is the input of func
    :param leave_cpu_num: num of cpu that not use
    :return: [return_of_func(list_inputs[0]), return_of_func(list_inputs[1]), ...]
    """
    cpu_cores = mp.cpu_count() - leave_cpu_num
    pool = mp.Pool(processes=cpu_cores)
    list_outputs = pool.map(func, list_inputs)
    pool.close()
    return list_outputs


def abstract_connected_regions(sample_list, aspect='rim', show=False, batch_size_cpu=32, batch_size_gpu=64, outer=False):
    """
    :param sample_list: each sample should be a 2d binary numpy array with one semantic, 0 indicate not this semantic
    and 1 means positive. If a sample has many semantics, you should slice it to only one semantic
    :param aspect: is the trait of each connected region, one of ['area', 'rim', 'both']
    :param show: whether print out the results during processing
    :param batch_size_cpu: batch size of multi-processing
    :param batch_size_gpu: batch size during get rim
    :param outer: if True, return the outer rim
    :return: a list of lists, which is the return of rim_length_and_id or area_and_id
    element of return list_info:
    [return_array, id_length(area)_dict, id_loc_dict]
    return_array has shape [a, b, 2], first channel is the length(area) map, second is the id map.
    """
    assert aspect == 'area' or aspect == 'rim' or aspect == 'both'
    assert len(np.shape(sample_list[0])) == 2
    num_samples = len(sample_list)
    sample_stack = np.stack(sample_list, axis=0)
    if show:
        print("there are", num_samples, "of samples")
        print("sample_stack has shape:", np.shape(sample_stack))
    rim_stack = np.zeros(np.shape(sample_stack), 'float32')
    for i in range(0, num_samples, batch_size_gpu):
        stop = min(num_samples, i + batch_size_gpu)
        rim_stack[i: stop, :, :] = get_rim(sample_stack[i: stop, :, :], outer=outer)
    leave_cpu_num = mp.cpu_count() - batch_size_cpu
    input_list = []
    if aspect == 'rim':
        for i in range(num_samples):
            func_input = [rim_stack[i, :, :], np.where(rim_stack[i, :, :] > epsilon)]
            input_list.append(func_input)
        return func_parallel(rim_length_and_id, input_list, leave_cpu_num)
    if aspect == 'area':
        for i in range(num_samples):
            func_input = [sample_stack[i, :, :], np.where(rim_stack[i, :, :] > epsilon)]
            input_list.append(func_input)
        return func_parallel(area_and_id, input_list, leave_cpu_num)
    if aspect == 'both':
        for i in range(num_samples):
            func_input = [rim_stack[i, :, :], np.where(rim_stack[i, :, :] > epsilon)]
            input_list.append(func_input)
        list_rim_info = func_parallel(rim_length_and_id, input_list, leave_cpu_num)
        for i in range(num_samples):
            input_list[i][0] = sample_stack[i, :, :]
        list_area_info = func_parallel(area_and_id, input_list, leave_cpu_num)
        return list_rim_info, list_area_info


def get_rim(input_array, outer=False):
    # the input_array is a image with shape [batch_size, a, b] or [a, b]
    # the output is a np array with shape [batch_size, a, b] or [a, b], 1 means rim points, 0 means not rim.
    global convolution_layer
    shape = np.shape(input_array)
    if len(shape) == 2:
        array = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0)
    elif len(shape) == 3:
        array = torch.from_numpy(input_array).unsqueeze(1)
    else:
        raise DimensionError(input_array)
    # now the array in shape [batch_size, 1, a, b]
    rim = convolution_layer(array.cuda())
    rim = rim.to('cpu')
    rim = rim.data.numpy()
    if outer:
        rim = np.array(rim < -epsilon, 'float32')
    else:
        rim = np.array(rim > epsilon, 'float32')
    if len(shape) == 2:
        return rim[0, 0, :, :]  # [a, b]
    else:
        return rim[:, 0, :, :]  # [batch_size, a, b]


def rim_length_and_id(func_input):
    # func_input = [input_rim, rim_points]
    # the input_rim is binary with shape [a, b], shows the rim of a semantic: 0, not rim, 1 rim.
    # rim_points = np.where(input_rim > epsilon), avoid multi-processing conflicts.
    # the output is quite similar to the input, except the value of rims become their length and the rim id.
    # the return_array has shape [a, b, 2], [:, :, 0] for length, [:, :, 1] for rim_id (1, 2, 3, ...)
    # output the dictionary for rim length and the locations
    input_rim = func_input[0]
    rim_points = func_input[1]
    a, b = np.shape(input_rim)
    return_array = np.zeros([a, b, 2], 'float32')
    # the first channel is the length, the second is the disconnected rim id.
    return_array[:, :, 0] = -input_rim
    # initially, the length is set to -1 for all rim points, and 0 for all not rim points.
    num_rim_points = len(rim_points[0])
    id_length_dict = {}
    id_loc_dict = {}

    rim_id = 0  # the rim_id counts for the disconnected rims.
    for index in range(num_rim_points):
        if return_array[rim_points[0][index], rim_points[1][index], 0] > 0:
            # this means this rim point has been allocated length
            continue
        else:
            # this mean this rim point is the first time we meet
            rim_id += 1  # the id is 1, 2, 3, ...
            length, rim_locations = broadcast_connected_component(return_array,
                                                                  (rim_points[0][index], rim_points[1][index]), rim_id)
            # now, the length and id has been broadcast to this connected component.
            id_length_dict[rim_id] = length
            id_loc_dict[rim_id] = rim_locations
    return return_array, id_length_dict, id_loc_dict


def area_and_id(func_input):
    # func_input = [input_sample, inner_rim_points]
    # detect connected component and stat there area and give them ids
    # the input_sample is binary with shape [a, b], 0, not semantic, 1 semantic.
    # inner_rim_points = np.where(inner_rim > epsilon), avoid multi-processing conflicts and reduce searching range.
    # the output is quite similar to the input, except the value of positives become their area and the are_id.
    # the return_array has shape [a, b, 2], [:, :, 0] for area, [:, :, 1] for are_id (1, 2, 3, ...)
    # output the dictionary for area and the locations
    input_sample = func_input[0]
    inner_rim_points = func_input[1]
    a, b = np.shape(input_sample)
    return_array = np.zeros([a, b, 2], 'float32')
    # the first channel is the length, the second is the disconnected rim id.
    return_array[:, :, 0] = -input_sample
    # initially, the length is set to -1 for all rim points, and 0 for all not rim points.
    num_rim_points = len(inner_rim_points[0])
    id_area_dict = {}
    id_loc_dict = {}

    area_id = 0  # the rim_id counts for the disconnected rims.
    for index in range(num_rim_points):
        if return_array[inner_rim_points[0][index], inner_rim_points[1][index], 0] > 0:
            # this means this rim point has been allocated length
            continue
        else:
            # this mean this rim point is the first time we meet
            area_id += 1  # the id is 1, 2, 3, ...
            area, area_locations = broadcast_connected_component(return_array,
                                                                  (inner_rim_points[0][index], inner_rim_points[1][index]), area_id)
            # now, the length and id has been broadcast to this connected component.
            id_area_dict[area_id] = area
            id_loc_dict[area_id] = area_locations
    return return_array, id_area_dict, id_loc_dict


def broadcast_connected_component(return_array, initial_location, component_id):
    # return_array has shape [a, b, 2]
    # initial_location is a tuple, (x, y)
    # return the number of pixels of this connected component and the location list like [(389, 401), (389, 402), ..].
    num_pixels = 0  # the num of pixels of this connected component
    un_labeled_rim = [initial_location, ]
    return_array[initial_location[0], initial_location[1], 1] = component_id
    component_locations = []
    while un_labeled_rim:  # this mean un_labeled_rim is not empty
        location = un_labeled_rim.pop()

        component_locations.append(location)  # get the locations of the connected component
        num_pixels += 1

        if return_array[location[0] + 1, location[1], 0] < -epsilon:  # search for the next unlabeled, rim pixel
            if not return_array[location[0] + 1, location[1], 1] == component_id:
                un_labeled_rim.append((location[0] + 1, location[1]))
                return_array[location[0] + 1, location[1], 1] = component_id  # label this unlabeled pixel

        if return_array[location[0] - 1, location[1], 0] < -epsilon:
            if not return_array[location[0] - 1, location[1], 1] == component_id:
                un_labeled_rim.append((location[0] - 1, location[1]))
                return_array[location[0] - 1, location[1], 1] = component_id

        if adjacency == 'loose' and return_array[location[0] + 1, location[1] + 1, 0] < -epsilon:
            if not return_array[location[0] + 1, location[1] + 1, 1] == component_id:
                un_labeled_rim.append((location[0] + 1, location[1] + 1))
                return_array[location[0] + 1, location[1] + 1, 1] = component_id

        if adjacency == 'loose' and return_array[location[0] + 1, location[1] - 1, 0] < -epsilon:
            if not return_array[location[0] + 1, location[1] - 1, 1] == component_id:
                un_labeled_rim.append((location[0] + 1, location[1] - 1))
                return_array[location[0] + 1, location[1] - 1, 1] = component_id

        if return_array[location[0], location[1] + 1, 0] < -epsilon:
            if not return_array[location[0], location[1] + 1, 1] == component_id:
                un_labeled_rim.append((location[0], location[1] + 1))
                return_array[location[0], location[1] + 1, 1] = component_id

        if return_array[location[0], location[1] - 1, 0] < -epsilon:
            if not return_array[location[0], location[1] - 1, 1] == component_id:
                un_labeled_rim.append((location[0], location[1] - 1))
                return_array[location[0], location[1] - 1, 1] = component_id

        if adjacency == 'loose' and return_array[location[0] - 1, location[1] - 1, 0] < -epsilon:
            if not return_array[location[0] - 1, location[1] - 1, 1] == component_id:
                un_labeled_rim.append((location[0] - 1, location[1] - 1))
                return_array[location[0] - 1, location[1] - 1, 1] = component_id

        if adjacency == 'loose' and return_array[location[0] - 1, location[1] + 1, 0] < -epsilon:
            if not return_array[location[0] - 1, location[1] + 1, 1] == component_id:
                un_labeled_rim.append((location[0] - 1, location[1] + 1))
                return_array[location[0] - 1, location[1] + 1, 1] = component_id

    for location in component_locations:
        return_array[location[0], location[1], 0] = num_pixels
    # print('this component has id', rim_id, 'length', length)
    return num_pixels, component_locations


def sort_on_id_loc_dict(id_loc_dict, id_volume_dict=None):
    # keys should be 1, 2, 3, ...
    # refactor the key of the connected_components big to small according to len(id_loc_dict[key])
    keys_list = list(id_loc_dict.keys())
    number_keys = len(keys_list)
    if id_volume_dict is None:
        id_volume_dict = {}
        for i in range(1, number_keys + 1):
            id_volume_dict[i] = len(id_loc_dict[i])
    old_factor_list = []
    for i in range(1, number_keys + 1):
        old_factor_list.append((i, id_volume_dict[i]))

    def adjacency_cmp(tuple_a, tuple_b):
        return tuple_a[1] - tuple_b[1]

    from functools import cmp_to_key
    old_factor_list.sort(key=cmp_to_key(adjacency_cmp), reverse=True)

    id_loc_dict_sorted = {}
    id_volume_dict_sorted = {}
    for i in range(0, number_keys):
        id_loc_dict_sorted[i + 1] = id_loc_dict[old_factor_list[i][0]]
        id_volume_dict_sorted[i + 1] = id_volume_dict[old_factor_list[i][0]]
    return id_loc_dict_sorted, id_volume_dict_sorted


def get_connect_region_2d(array, strict=True, sort=True, get_return_array=False):
    # the return_array has shape [a, b, 2], [:, :, 0] for area, [:, :, 1] for are_id (1, 2, 3, ...)
    assert len(np.shape(array)) == 2
    global adjacency, convolution_layer
    if adjacency == 'strict' and strict is not True:
        convolution_layer = GetRimLoose()
        convolution_layer = convolution_layer.cuda()
        adjacency = 'loose'
        if torch.cuda.device_count() > 1:
            convolution_layer = nn.DataParallel(convolution_layer)
    if adjacency == 'loose' and strict is True:
        convolution_layer = GetRimStrict()
        convolution_layer = convolution_layer.cuda()
        adjacency = 'strict'
        if torch.cuda.device_count() > 1:
            convolution_layer = nn.DataParallel(convolution_layer)
    array_rim = get_rim(array, False)
    return_array, id_area_dict, id_loc_dict = area_and_id([array, np.where(array_rim > 0.5)])
    # Functions.image_show(return_array[:, :, 0])  # indicate the area
    # Functions.image_show(return_array[:, :, 1])  # indicate the connect_component id
    if get_return_array:
        if sort:
            return sort_on_id_loc_dict(id_loc_dict, id_area_dict)[0], return_array
        else:
            return id_loc_dict, return_array
    if sort:
        return sort_on_id_loc_dict(id_loc_dict, id_area_dict)[0]
    else:
        return id_loc_dict


def update_frequency(id_scale_dict, frequency_array):
    key_list = list(id_scale_dict.keys())
    for key in key_list:
        scale = id_scale_dict[key]
        if scale >= 5000:
            scale = 4999
        frequency_array[scale] += 1


def abstract_data_set(top_dict, save_dict, gt_slice=3, scale=5000):
    # stat on the scale free property and save the results
    # gt_slice: the channel that the mask stored
    sample_name_list = os.listdir(top_dict)
    print("data-set dict", top_dict)
    print("save to", save_dict)
    print("there are", len(sample_name_list), "number of samples")

    for fold in range(20):
        print("fold:", fold)
        if os.path.exists(save_dict + 'frequency_rim.npy'):
            frequency_array_rim = np.load(save_dict + 'frequency_rim.npy')
            print("load rim frequency array")
        else:
            frequency_array_rim = np.zeros([scale, ], 'int32')
            print("new rim frequency array")

        if os.path.exists(save_dict + 'frequency_area.npy'):
            frequency_array_area = np.load(save_dict + 'frequency_area.npy')
            print("load area frequency array")
        else:
            frequency_array_area = np.zeros([scale, ], 'int32')
            print("new area frequency array")

        sample_list = []
        print("prepare sample list...")
        for name in sample_name_list[fold::20]:
            sample_list.append(np.array(np.load(top_dict + name)[:, :, gt_slice] > 0.5, 'float32'))
        print("abstracting information....")
        rim_info_list, area_info_list = abstract_connected_regions(sample_list, 'both', True)

        print("update rim frequencies")
        for rim_info in rim_info_list:
            update_frequency(rim_info[1], frequency_array_rim)
        print("update area frequencies")
        for area_info in area_info_list:
            update_frequency(area_info[1], frequency_array_area)

        Functions.save_np_array(save_dict, 'frequency_rim.npy', frequency_array_rim)
        Functions.save_np_array(save_dict, 'frequency_area.npy', frequency_array_area)
        print("fold:", fold)
        print(frequency_array_rim[1: 50])
        print(frequency_array_area[1: 50])


if __name__ == '__main__':
    exit()
    print(adjacency)
    abstract_data_set('/ibex/scratch/projects/c2052/COVID-19/2D_Model/datasets/train_dir/',
                      '/ibex/scratch/projects/c2052/prognosis_project/scale_free_property/infection/Z/', gt_slice=1)

    abstract_data_set('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/Z/',
                      '/ibex/scratch/projects/c2052/prognosis_project/scale_free_property/air_way/Z/')
    abstract_data_set('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/X/',
                      '/ibex/scratch/projects/c2052/prognosis_project/scale_free_property/air_way/X/')
    abstract_data_set('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/Y/',
                      '/ibex/scratch/projects/c2052/prognosis_project/scale_free_property/air_way/Y/')

    abstract_data_set('/ibex/scratch/projects/c2052/blood_vessel_seg/training_samples/X/',
                      '/ibex/scratch/projects/c2052/prognosis_project/scale_free_property/blood_vessel/X/')
    abstract_data_set('/ibex/scratch/projects/c2052/blood_vessel_seg/training_samples/Y/',
                      '/ibex/scratch/projects/c2052/prognosis_project/scale_free_property/blood_vessel/Y/')
    abstract_data_set('/ibex/scratch/projects/c2052/blood_vessel_seg/training_samples/Z/',
                      '/ibex/scratch/projects/c2052/prognosis_project/scale_free_property/blood_vessel/Z/')

