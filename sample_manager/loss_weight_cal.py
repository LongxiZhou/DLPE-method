import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Tool_Functions.Functions as Functions
import os
import math
import random

np.set_printoptions(precision=10, suppress=True)
epsilon = 0.001


class GetRim(nn.Module):
    def __init__(self):
        super(GetRim, self).__init__()
        super().__init__()
        kernel = [[[[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]]]
        kernel = torch.FloatTensor(kernel)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=1)
        return x


convolution_layer = GetRim()
convolution_layer = convolution_layer.cuda()
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    convolution_layer = nn.DataParallel(convolution_layer)
else:
    print("Using only single GPU")


def get_rim(input_array, outer=False):
    # the input_array is a image with shape [a, b]
    # the output is a np array with shape [a, b], 1 means rim points, 0 means not rim.
    global convolution_layer
    array = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0)
    rim = convolution_layer(array.cuda())
    rim = rim.to('cpu')
    rim = rim.data.numpy()
    if outer:
        rim = np.array(rim < -epsilon, 'float32')
    else:
        rim = np.array(rim > epsilon, 'float32')
    return rim[0, 0, :, :]


def rim_length_and_id(input_rim):
    # the input_rim is binary with shape [a, b], shows the rim of the segmentation
    # the output is quite similar to the input, except the value of rims become their length and the rim id.
    # the return_array has shape [a, b, 2], [:, :, 0] for length, [:, :, 1] for rim_id (1, 2, 3, ...)
    # output the dictionary for rim length and the locations
    a, b = np.shape(input_rim)
    return_array = np.zeros([a, b, 2], 'float32')
    # the first channel is the length, the second is the disconnected rim id.
    return_array[:, :, 0] = -input_rim
    # initially, the length is set to -1 for all rim points, and 0 for all not rim points.

    rim_points = np.where(return_array[:, :, 0] < -epsilon)
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


def broadcast_connected_component(return_array, initial_location, rim_id):
    # return_array has shape [a, b, 2]
    # initial_location is a tuple, (x, y)
    # return the length of this border (int) and the location list like [(389, 401), (389, 402), ..].
    length = 0  # the length of this connected component
    un_labeled_rim = [initial_location, ]
    return_array[initial_location[0], initial_location[1], 1] = rim_id
    rim_locations = []
    while un_labeled_rim:  # this mean un_labeled_rim is not empty
        location = un_labeled_rim.pop()

        rim_locations.append(location)  # get the locations of the connected component
        length += 1

        if return_array[location[0] + 1, location[1], 0] < -epsilon:  # search for the next unlabeled, rim pixel
            if not return_array[location[0] + 1, location[1], 1] == rim_id:
                un_labeled_rim.append((location[0] + 1, location[1]))
                return_array[location[0] + 1, location[1], 1] = rim_id  # label this unlabeled pixel

        if return_array[location[0] - 1, location[1], 0] < -epsilon:
            if not return_array[location[0] - 1, location[1], 1] == rim_id:
                un_labeled_rim.append((location[0] - 1, location[1]))
                return_array[location[0] - 1, location[1], 1] = rim_id

        if return_array[location[0] + 1, location[1] + 1, 0] < -epsilon:
            if not return_array[location[0] + 1, location[1] + 1, 1] == rim_id:
                un_labeled_rim.append((location[0] + 1, location[1] + 1))
                return_array[location[0] + 1, location[1] + 1, 1] = rim_id

        if return_array[location[0] + 1, location[1] - 1, 0] < -epsilon:
            if not return_array[location[0] + 1, location[1] - 1, 1] == rim_id:
                un_labeled_rim.append((location[0] + 1, location[1] - 1))
                return_array[location[0] + 1, location[1] - 1, 1] = rim_id

        if return_array[location[0], location[1] + 1, 0] < -epsilon:
            if not return_array[location[0], location[1] + 1, 1] == rim_id:
                un_labeled_rim.append((location[0], location[1] + 1))
                return_array[location[0], location[1] + 1, 1] = rim_id

        if return_array[location[0], location[1] - 1, 0] < -epsilon:
            if not return_array[location[0], location[1] - 1, 1] == rim_id:
                un_labeled_rim.append((location[0], location[1] - 1))
                return_array[location[0], location[1] - 1, 1] = rim_id

        if return_array[location[0] - 1, location[1] - 1, 0] < -epsilon:
            if not return_array[location[0] - 1, location[1] - 1, 1] == rim_id:
                un_labeled_rim.append((location[0] - 1, location[1] - 1))
                return_array[location[0] - 1, location[1] - 1, 1] = rim_id

        if return_array[location[0] - 1, location[1] + 1, 0] < -epsilon:
            if not return_array[location[0] - 1, location[1] + 1, 1] == rim_id:
                un_labeled_rim.append((location[0] - 1, location[1] + 1))
                return_array[location[0] - 1, location[1] + 1, 1] = rim_id

    for location in rim_locations:
        return_array[location[0], location[1], 0] = length
    # print('this component has id', rim_id, 'length', length)
    return length, rim_locations


def get_connected_region_and_energy(input_array, length_and_id_array, length_dict):
    # the input_array is binary with shape [a, b], lesion: 1 and normal: 0
    # length_and_id_array is float32 with shape [a, b, 2], first channel is the length of border, second is their id
    # from length_dict and location_dict we know the length and the locations of each connected borders
    # return a float32 array with shape [a, b, 2], first channel is the energy of each pixel, second is their id
    lesions = np.where(input_array > epsilon)
    num_lesion_pixel = len(lesions[0])
    lesion_energy_dict = {}  # key: lesion_id, like 1, 2, 3, ...; value: energy, a float > 0
    lesion_location_dict = {}  # key: lesion_id, like 1, 2, 3, ...; value: list, elements are (x, y)
    lesion_border_dict = {}  # key: lesion_id, like 1, 2, 3, ...; value: list, elements are int indication border keys
    a, b = np.shape(input_array)
    energy_id_array = np.zeros([a, b, 2], 'float32')
    energy_id_array[:, :, 0] = -input_array
    region_id = 0
    for index in range(num_lesion_pixel):
        if energy_id_array[lesions[0][index], lesions[1][index], 0] > 0:
            # this means this region has been visited
            continue
        else:  # this means we first time visit this connected region
            region_id += 1
            energy, locations, border_keys = broadcast_connected_region(energy_id_array,
                                                                        (lesions[0][index], lesions[1][index]),
                                                                        region_id, length_and_id_array, length_dict)
            # now, the energy and id has been broadcast to this connected region.
            lesion_energy_dict[region_id] = energy
            lesion_location_dict[region_id] = locations
            lesion_border_dict[region_id] = border_keys
    return energy_id_array, lesion_energy_dict, lesion_location_dict, lesion_border_dict


def broadcast_connected_region(energy_id_array, initial_location, region_id, length_and_id_array, length_dict):
    # energy_id_array has shape [a, b, 2]
    # initial_location is a tuple, (x, y), region_id is a int like 1, 2, 3, 4
    # length_and_id_array has shape [a, b, 2], length_dict indicate the length of each connected border
    # return the energy of this region and the location list like [(389, 401), (389, 402), ..].
    energy = 0  # the energy of this connected region
    un_labeled_region = [initial_location, ]
    energy_id_array[initial_location[0], initial_location[1], 1] = region_id
    region_locations = []
    border_keys = []
    while un_labeled_region:  # this mean un_labeled_region is not empty
        location = un_labeled_region.pop()
        region_locations.append(location)

        # we check the border energy
        right = length_and_id_array[location[0] + 1, location[1], 1]
        if right > 0:  # means this is a border pixel
            if right not in border_keys:
                border_keys.append(int(right))
                energy += length_dict[right]

        left = length_and_id_array[location[0] - 1, location[1], 1]
        if left > 0:  # means this is a border pixel
            if left not in border_keys:
                border_keys.append(int(left))
                energy += length_dict[left]

        up = length_and_id_array[location[0], location[1] + 1, 1]
        if up > 0:  # means this is a border pixel
            if up not in border_keys:
                border_keys.append(int(up))
                energy += length_dict[up]

        down = length_and_id_array[location[0], location[1] - 1, 1]
        if down > 0:  # means this is a border pixel
            if down not in border_keys:
                border_keys.append(int(down))
                energy += length_dict[down]

        # broadcast to the adjacent pixels
        if energy_id_array[location[0] + 1, location[1], 0] < -epsilon:  # search for the next unlabeled, region pixel
            if not energy_id_array[location[0] + 1, location[1], 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1]))
                energy_id_array[location[0] + 1, location[1], 1] = region_id  # label this unlabeled pixel

        if energy_id_array[location[0] - 1, location[1], 0] < -epsilon:  # search for the next unlabeled, region pixel
            if not energy_id_array[location[0] - 1, location[1], 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1]))
                energy_id_array[location[0] - 1, location[1], 1] = region_id  # label this unlabeled pixel

        if energy_id_array[location[0], location[1] + 1, 0] < -epsilon:  # search for the next unlabeled, region pixel
            if not energy_id_array[location[0], location[1] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1))
                energy_id_array[location[0], location[1] + 1, 1] = region_id  # label this unlabeled pixel

        if energy_id_array[location[0], location[1] - 1, 0] < -epsilon:  # search for the next unlabeled, region pixel
            if not energy_id_array[location[0], location[1] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1))
                energy_id_array[location[0], location[1] - 1, 1] = region_id  # label this unlabeled pixel

    for location in region_locations:
        energy_id_array[location[0], location[1], 0] = energy
    border_keys = set(border_keys)
    # print('this region has id', region_id, 'energy', energy, 'the borders are', border_keys)
    return energy, region_locations, border_keys


def integrate_connected_region(lesion_energy_dict, lesion_location_dict, lesion_border_dict, length_dict):
    # the rim is the outer rim of the region, thus, it is possible that several disconnected region has the same border
    # this function, integrate disconnected regions into a connected region if they are so adjacent that share borders.
    # return new_energy_dict, new_location_dict, new_border_dict for the new integrated connected regions
    num_connected_old = len(lesion_energy_dict)
    num_new_connected = 1
    new_energy_dict = {}
    new_location_dict = {}
    new_border_dict = {}
    if num_connected_old > 0:
        new_location_dict[1] = lesion_location_dict[1]
        new_border_dict[1] = lesion_border_dict[1]
    for key_old in range(2, num_connected_old + 1):
        borders = lesion_border_dict[key_old]  # the border indices of this old connected region
        flag = 0
        for index in borders:
            if flag == 1:
                break
            for key_new in range(1, num_new_connected + 1):
                if index in new_border_dict[key_new]:  # this means old region (key_old) should be integrated into new region (key_new)
                    new_border_dict[key_new] = new_border_dict[key_new] | borders
                    new_location_dict[key_new] = new_location_dict[key_new] + lesion_location_dict[key_old]
                    flag = 1
                    break  # now this region is integrated into new region
        if flag == 0:  # this means old region (key_old) can not integrated into any new regions
            num_new_connected += 1
            new_location_dict[num_new_connected] = lesion_location_dict[key_old]
            new_border_dict[num_new_connected] = lesion_border_dict[key_old]
    for key_new in range(1, num_new_connected + 1):  # let us calculate the energy for the integrated regions
        energy = 0
        for index in new_border_dict[key_new]:
            energy += length_dict[index]
        new_energy_dict[key_new] = energy

    if num_new_connected == num_new_connected:  # this means the integration complete!
        return new_energy_dict, new_location_dict, new_border_dict
    else:
        return integrate_connected_region(new_energy_dict, new_location_dict, new_border_dict, length_dict)


def calculate_balance_weights(input_array, rim_energy_factor=1, connect_energy_factor=10, augment_index=1,
                              rim_enhance=0, area_enhance=0, border_outer=False, return_stat=False):
    # the input_array is binary with shape [a, b], lesion: 1 and normal: 0
    # output a weight map for lesion pixels, indicating the "distribution of loss" for every LESION pixel
    # border_outer, defines what is the rim. If false, the rim are inside lesions, otherwise outside lesions.
    # each connected lesion has an "energy", which is solely determined by the length of its contours.
    # energy_pixel = pow((len_contours * rim_energy_factor + num_connected * connect_energy_factor)/area, augment_index)
    # and then the energy is uniformly distributed to every pixels in this lesion. If we want the rim_pixel or
    # add extra energy to every pixels, use rim_enhance and area_enhance.
    # return_stat=True when cal_balance_weights_for_different_dimensions, it calculate the 0D, 1D, 2D losses.

    rim = get_rim(input_array, border_outer)
    length_and_id_array, length_dict, rim_location_dict = rim_length_and_id(rim)
    energy_id_array, lesion_energy_dict, lesion_location_dict, lesion_border_dict = \
        get_connected_region_and_energy(input_array, length_and_id_array, length_dict)

    if return_stat:  # only need to see how many
        total_connected_areas = len(lesion_energy_dict)  # 0D loss
        total_rim_length = np.sum(rim)  # 1D loss
        total_num_lesion = np.sum(input_array)  # 2D loss
        return total_connected_areas, total_rim_length, total_num_lesion

    new_energy_dict, new_location_dict, new_border_dict = \
        integrate_connected_region(lesion_energy_dict, lesion_location_dict, lesion_border_dict, length_dict)

    weight_array = np.zeros(np.shape(input_array), 'float32')
    num_integrated_regions = len(new_energy_dict)
    num_connected_regions = len(lesion_energy_dict)

    num_connected_region_in_integrated_dict = {}
    total_connected_number = 0
    for key_new in range(1, num_integrated_regions + 1):
        borders = new_border_dict[key_new]
        num_connected = 0  # the number of connected regions for this integrated region
        for region_index in range(1, num_connected_regions + 1):
            borders_of_connected_region = lesion_border_dict[region_index]
            if borders_of_connected_region.issubset(borders):
                num_connected += 1
                total_connected_number += 1
        num_connected_region_in_integrated_dict[key_new] = num_connected

    if not total_connected_number == num_connected_regions:
        print('total connected', total_connected_number, 'number_connected_regions', num_connected_regions)
        return None, 0

    if rim_enhance > 0:  # this means we further add contrast to the rims
        rim = get_rim(input_array, outer=False)  # only enhance inner rim
        length_and_id_array_inner, length_dict_inner, rim_location_dict_inner = rim_length_and_id(rim)
        num_connected_border = len(length_dict_inner)  # length_dict and location_dict has the same key, i.e. 1, 2, ...
        for index in range(1, num_connected_border + 1):
            locations = rim_location_dict_inner[index]
            for loc in locations:
                weight_array[loc[0], loc[1]] += rim_enhance

    if area_enhance > 0:  # this means we further add contrast to the lesions
        for key_new in range(1, num_integrated_regions + 1):
            locations = new_location_dict[key_new]
            for loc in locations:
                weight_array[loc[0], loc[1]] += area_enhance

    for key_new in range(1, num_integrated_regions + 1):
        locations = new_location_dict[key_new]
        area = len(locations)
        num_connected = num_connected_region_in_integrated_dict[key_new]
        energy = new_energy_dict[key_new] * rim_energy_factor + num_connected * connect_energy_factor
        energy_pixel = energy / area
        energy_pixel = math.pow(energy_pixel, augment_index)
        for loc in locations:
            weight_array[loc[0], loc[1]] += energy_pixel

    return weight_array, np.sum(weight_array)


def calculate_balance_weights_all_files(label_array_dict, energy_array, save_dict):
    """
    :param label_array_dict: contains arrays (compressed) with shape [length, width, classes]
    :param energy_array: an array containing energy factors for all classes
    :param save_dict: output the weight map
    :return: None
    """
    label_array_name_list = os.listdir(label_array_dict)
    print('we have:', len(label_array_name_list), 'label')
    processed_count = 0
    num_label_array = len(label_array_name_list)
    class_num = np.shape(energy_array)[0]
    print('the energy array is (first line normal):\n', energy_array)
    print('we have', 1, 'normal and', class_num - 1, 'types of lesions')

    if not label_array_dict[-1] == '/':
        label_array_dict = label_array_dict + '/'
    print('label_array from:', label_array_dict)

    # label_array = np.load(label_array_dict + label_array_name_list[0])['array']
    # array_shape = np.shape(label_array)  # we get the shape of the label array
    array_shape = [512, 512, 2]

    for label_array_name in label_array_name_list:
        # TODO: change training sample into label array
        label_array = np.zeros(array_shape, 'float32')
        label_array[:, :, 1] = np.load(label_array_dict + label_array_name)[:, :, 3]
        label_array[:, :, 0] = 1 - label_array[:, :, 1]
        weight_array = np.zeros(array_shape, 'float32')
        for semantic in range(1, class_num):
            weight_map, _ = calculate_balance_weights(
                label_array[:, :, semantic], connect_energy_factor=energy_array[semantic][0], rim_energy_factor=energy_array[semantic][1],
                rim_enhance=energy_array[semantic][3], area_enhance=energy_array[semantic][2])
            if weight_map is None:
                break
            weight_array[:, :, 0] += get_rim(label_array[:, :, semantic], outer=True) * energy_array[semantic][1]
            weight_array[:, :, semantic] = weight_map
        if weight_map is not None:
            weight_array[:, :, 0] += energy_array[0][2] * label_array[:, :, 0] * 3 / 4
            Functions.save_np_array(save_dict, 'weights_' + label_array_name[0:-4], weight_array)
        else:
            print('sample:', label_array_name, 'is strange')
        processed_count += 1
        if processed_count % 50 == 0:
            print('processed', processed_count, ', left:', num_label_array - processed_count)


def cal_energy_parameters_for_one_channel(sub_sample_label_dict, channel, importance=1):
    """
    the loss comes equally from four sources: connected component (0D), boundary (1D), area (2D), and rim_enhance
    e.g. a small region with long boundaries means it accounts for lots of 1D loss and little of 2D loss.
    If border_outer=False, boundaries are inside lesions, and all connected regions account for the same 0D loss
    If border_outer=True, boundaries are outside lesions, 0D loss are the same inside the same integrated connected
    region determined by the outer boundaries.
    0D, 1D, 2D loss are uniformly distributed into every pixels inside the integrated connected region;
    rim_enhance is then added to the boundary pixels.
    :param sub_sample_label_dict: The dict of representative training sample labels. This function calculate how to
    balance the loss weights according to these training sample labels. Training sample labels should be numpy arrays
    shaped: [length, width, channel], and when the channel is specified, it should be a binary image, with 1 means
    positive, it can be a probability [0, 1]. When summing all channels, we get a 2D array of all ones.
    :param channel: which channel we need to calculate? The weights are calculated channel-wise. The theoretical basis
    is that, some TYPES lesions are big and folded; while some are small and smooth. When doing performance measure, we
    don't care about this difference in the classes. Thus, different classes should account the same training loss.
    Channel 0 is the probability mask for normal pixels.
    :param importance: There may be a special class is extremely important. Increase the importance will increase the
    proportion of training loss for this class.
    :return: connect_energy_factor, rim_energy_factor, area_enhance, rim_enhance
                      0D                     1D              2D
    """
    sample_names_list = os.listdir(sub_sample_label_dict)
    total_num_connected_areas = 0  # the number of connected areas in this sub set, counts for 0D loss
    total_rim_length = 0  # the number of rim voxels, counts for 1D loss
    total_lesion_area = 0  # the number of lesion voxels, counts for 2D loss

    if not sub_sample_label_dict[-1] == '/':
        sub_sample_label_dict = sub_sample_label_dict + '/'
    for sample_name in sample_names_list:
        sample = np.load(sub_sample_label_dict + sample_name)  # sample should in [width, length, channel]
        mask = sample[:, :, channel]
        num_connected_areas, num_rim_voxels, num_lesion_voxels = calculate_balance_weights(mask, return_stat=True)

        total_num_connected_areas += num_connected_areas
        total_rim_length += num_rim_voxels
        total_lesion_area += num_lesion_voxels

    num_samples = len(sample_names_list)
    num_loss_per_dimension = num_samples * importance
    # each sample and each class is defaulted to have 4 units of losses: 3 units, 0D, 1D, 2D, which distributed
    # uniformly on lesions; and one unit distributed uniformly on the rim pixels.
    area_enhance = num_loss_per_dimension / total_lesion_area  # thus, averagely each slice 1 units of 2D loss
    rim_energy_factor = num_loss_per_dimension / total_rim_length  # thus, averagely each slice 1 units of 1D loss
    connect_energy_factor = num_loss_per_dimension / total_num_connected_areas  # each slice 1 units of 0D loss
    rim_enhance = num_loss_per_dimension / total_rim_length  # averagely further add 1 units to enhance the rim pixels

    return connect_energy_factor, rim_energy_factor, area_enhance, rim_enhance


# different channels have different energy parameters for 0D, 1D, 2D. And different channels have different importance.
connect_energy_factor_channel_wise = []  # energy factor for 0D
rim_energy_factor_channel_wise = []  # energy factor for 1D
area_enhance_channel_wise = []  # energy factor for 2D
rim_enhance_channel_wise = []  # further enhance rim
# with the above energy factors, each class and each dimensions will has the same attention (loss)
# then, use calculate_balance_weights(), to correctly distribute these energy (attention) into every voxel.
importance_list = None  # None for defaulted, each channel and each sample has 4 units of loss.


# if importance_list = [1, 2, 1, 0], which means channel 0 & 3 have 4 units per sample, channel 1 has 8, channel 3 has 0
# that means the model will not optimize predictions on channel 3, and pay more attention to channel 1.


def cal_energy_parameters_for_all_channels(sub_sample_label_dict, save_dict):
    # this is the integrated version of cal_energy_parameters_for_one_channel.
    # sample_label should have shape [width, length, channels_for_semantics]
    # e.g. in our IEEE TMI, the channels_for_semantics is 2: normal and infection.
    # if importance_list=None, then all classes (channels) has the same importance
    # save_dict is for saving the connect_energy_factor_channel_wise, rim_energy_factor_channel_wise, ...
    global connect_energy_factor_channel_wise, rim_energy_factor_channel_wise, area_enhance_channel_wise, \
        rim_enhance_channel_wise, importance_list

    connect_energy_factor_channel_wise = [0]
    rim_energy_factor_channel_wise = [0]
    area_enhance_channel_wise = []
    rim_enhance_channel_wise = [0]

    if not sub_sample_label_dict[-1] == '/':
        sub_sample_label_dict = sub_sample_label_dict + '/'
    sample_names_list = os.listdir(sub_sample_label_dict)

    if sample_names_list[0][-1] == 'z':
        compressed = True
    else:
        compressed = False

    if not compressed:
        sample_first = np.load(sub_sample_label_dict + sample_names_list[0])
    else:
        sample_first = np.load(sub_sample_label_dict + sample_names_list[0])['array']

    num_channels = np.shape(sample_first)[2]  # sample has shape [width, length, channel]
    print('we have', num_channels, 'classes')
    if importance_list is None:  # this means model should pay equal attention to all channels
        importance_list = list(np.ones([num_channels], 'float32'))
        print('using defaulted importance list')
    else:
        print('the importance list is:', importance_list)

    total_num_connected_areas_list = list(np.zeros([num_channels], 'float32'))  # counts for size of 0D component
    total_rim_length_list = list(np.zeros([num_channels], 'float32'))  # counts for size of 1D component
    total_lesion_area_list = list(np.zeros([num_channels], 'float32'))  # counts for size of 2D component

    total_num_connected_areas_list[0] = 0
    total_rim_length_list[0] = 0  # we only calculate the area of normal pixels.

    # now we stat the area, length and connected component numbers in the data set
    num_samples = len(sample_names_list)
    num_processed = 0
    for sample_name in sample_names_list:
        if not compressed:
            sample = np.load(sub_sample_label_dict + sample_name)
        else:
            sample = np.load(sub_sample_label_dict + sample_name)['array']
        total_normal_voxel = np.sum(sample[:, :, 0])  # channel 0 is the normal voxel mask!
        total_lesion_area_list[0] += total_normal_voxel  # only 2D loss for normal voxels!

        for channel in range(1, num_channels):
            mask = sample[:, :, channel]
            num_connected_areas, num_rim_voxels, num_lesion_voxels = calculate_balance_weights(mask, return_stat=True)

            total_num_connected_areas_list[channel] += num_connected_areas  # 0D
            total_rim_length_list[channel] += num_rim_voxels  # 1D
            total_lesion_area_list[channel] += num_lesion_voxels  # 2D
        num_processed += 1
        if num_processed % 100 == 0:
            print('processed', num_processed, 'of total', num_samples)

    # now we calculate how to distribute the loss to every voxel.
    num_loss_normal = num_samples * importance_list[0] * 4
    area_enhance_channel_wise.append(num_loss_normal / total_lesion_area_list[0])  # these lines for normal voxels
    print('the balance weight for normal voxels is:', area_enhance_channel_wise[0], '\n')

    for channel in range(1, num_channels):  # these for lesions.
        num_loss_per_dimension = num_samples * importance_list[channel]
        # each sample and each class is defaulted to have 4 units of losses: 3 units, 0D, 1D, 2D, which distributed
        # uniformly on lesions; and one unit distributed uniformly on the rim pixels.
        area_enhance_channel_wise.append(num_loss_per_dimension / total_lesion_area_list[channel])
        rim_energy_factor_channel_wise.append(num_loss_per_dimension / total_rim_length_list[channel])
        connect_energy_factor_channel_wise.append(num_loss_per_dimension / total_num_connected_areas_list[channel])
        rim_enhance_channel_wise.append(num_loss_per_dimension / total_rim_length_list[channel])
        print('lesion class', channel, '\'s reports:')
        print('the total number of (connected_regions, rim_pixels, lesion_pixels) is:',
              (
              total_num_connected_areas_list[channel], total_rim_length_list[channel], total_lesion_area_list[channel]))
        print('rim pixels will be enhanced by:', rim_enhance_channel_wise[channel])
        print('all lesion pixels will be further enhanced by:', area_enhance_channel_wise[channel])

    energy_parameters_array = np.zeros([num_channels, 4], 'float32')
    energy_parameters_array[:, 0] = np.array(connect_energy_factor_channel_wise, 'float32')  # 0D
    energy_parameters_array[:, 1] = np.array(rim_energy_factor_channel_wise, 'float32')  # 1D
    energy_parameters_array[:, 2] = np.array(area_enhance_channel_wise, 'float32')  # 2D
    energy_parameters_array[:, 3] = np.array(rim_enhance_channel_wise, 'float32')  # further enhance rim

    print('\n\nthe energy_parameters for normal(first line), and the lesions:\n', energy_parameters_array)
    if not save_dict[-1] == '/':
        save_dict = save_dict + '/'
    Functions.save_np_array(save_dict, 'energy_parameters_array.npy', energy_parameters_array, False)
    print('energy_parameters_array has been saved as:', save_dict + 'energy_parameters_array.npy')

    component_stat_array = np.zeros([num_channels, 3], 'float32')
    component_stat_array[:, 0] = np.array(total_num_connected_areas_list, 'float32')  # 0D
    component_stat_array[:, 1] = np.array(total_rim_length_list, 'float32')  # 1D
    component_stat_array[:, 2] = np.array(total_lesion_area_list, 'float32')  # 2D

    Functions.save_np_array(save_dict, 'component_stat_array.npy', component_stat_array, False)
    print('\n\nthe component_stat_array lines are: 0D, 1D, 2D, rim\n', energy_parameters_array)


def convert_training_samples_into_label_array_subset(training_sample_dict, channels_for_raw_data, save_dict,
                                                     subset_ratio=0.05):
    """
    This function, prepare subset for the one-hot label arrays, then we can calculate the energy parameters.
    :param training_sample_dict: contains the training samples of our model
    Training samples are 2D segmentation samples.
    Training samples should have shape [width, length, channels_for_raw_data + channels_for_semantics]
    e.g, in our IEEE TMI, there are 5 channels_for_raw_data: -5, -2, 0, +2, +5, and 2 channels for semantics: normal, infection
    :param channels_for_raw_data: np.shape(training_sample) = [width, length, channels_for_raw_data + channels_for_semantics]
    :param save_dict: the dict that saving the label arrays
    :param subset_ratio: the ratio of selecting training samples to calculate the energy_parameters.
    :return: the number of label arrays
    """
    if not training_sample_dict[-1] == '/':
        training_sample_dict = training_sample_dict + '/'
    sample_name_list = os.listdir(training_sample_dict)
    num_training_samples = len(sample_name_list)
    random.shuffle(sample_name_list)
    num_selected = int(num_training_samples * subset_ratio)
    selected_name_list = sample_name_list[0: num_selected]
    print('we will convert:', num_selected, 'training samples.')
    # now we know which files will be used for cal the energy parameters.

    training_sample = np.load(training_sample_dict + selected_name_list[0])
    training_sample_shape = np.shape(training_sample)
    print('the training samples has shape:', training_sample_shape)
    channel_num = training_sample_shape[2]
    print('the training sample has:', channels_for_raw_data, 'channels for raw data')
    assert channels_for_raw_data > 0

    print('we will save the label arrays into:', save_dict)

    converted_count = 0
    for selected_name in selected_name_list:
        training_sample = np.load(training_sample_dict + selected_name)
        if channel_num - channels_for_raw_data > 1:
            label_array = training_sample[:, :, channels_for_raw_data: channel_num]
        else:  # this means there is only one channel indicating positives
            label_array = np.zeros((training_sample_shape[0], training_sample_shape[1], 2), 'float32')
            label_array[:, :, 1] = training_sample[:, :, channel_num - 1]
            label_array[:, :, 0] = 1 - label_array[:, :, 1]
        Functions.save_np_array(save_dict, 'label_' + selected_name[0:-4], label_array, True)
        converted_count += 1
        if converted_count % 100 == 0:
            print('processed', converted_count, 'number of samples, total', num_selected)

    return num_selected


#def cal_all_balance_weight_file(root_dict)


energy_x = np.load('/ibex/scratch/projects/c2052/air_tube_seg/reports/X/energy_parameters_array.npy')
energy_y = np.load('/ibex/scratch/projects/c2052/air_tube_seg/reports/Y/energy_parameters_array.npy')
energy_z = np.load('/ibex/scratch/projects/c2052/air_tube_seg/reports/Z/energy_parameters_array.npy')
print(energy_x)
print(energy_y)
print(energy_z)
calculate_balance_weights_all_files('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/X/', energy_x, '/ibex/scratch/projects/c2052/air_tube_seg/balance_weight_array_rim_enhance/X')
calculate_balance_weights_all_files('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/Y/', energy_y, '/ibex/scratch/projects/c2052/air_tube_seg/balance_weight_array_rim_enhance/Y')
calculate_balance_weights_all_files('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/Z/', energy_z, '/ibex/scratch/projects/c2052/air_tube_seg/balance_weight_array_rim_enhance/Z')



exit()

convert_training_samples_into_label_array_subset('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/X/', 3,
                                                 '/ibex/scratch/projects/c2052/air_tube_seg/label_array/X/',
                                                 subset_ratio=0.25)
cal_energy_parameters_for_all_channels('/ibex/scratch/projects/c2052/air_tube_seg/label_array/X/',
                                       '/ibex/scratch/projects/c2052/air_tube_seg/reports/X/')

convert_training_samples_into_label_array_subset('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/Y/', 3,
                                                 '/ibex/scratch/projects/c2052/air_tube_seg/label_array/Y/',
                                                 subset_ratio=0.25)
cal_energy_parameters_for_all_channels('/ibex/scratch/projects/c2052/air_tube_seg/label_array/Y/',
                                       '/ibex/scratch/projects/c2052/air_tube_seg/reports/Y/')

convert_training_samples_into_label_array_subset('/ibex/scratch/projects/c2052/air_tube_seg/training_samples/Z/', 3,
                                                 '/ibex/scratch/projects/c2052/air_tube_seg/label_array/Z/',
                                                 subset_ratio=0.25)
cal_energy_parameters_for_all_channels('/ibex/scratch/projects/c2052/air_tube_seg/label_array/Z/',
                                       '/ibex/scratch/projects/c2052/air_tube_seg/reports/Z/')
exit()

a = '/home/zhoul0a/Desktop/COVID-19/arrays_raw/xgfy-B000140_2020-02-20.npy'
b = '/home/zhoul0a/Desktop/air_tube_seg/arrays_raw/xwqg-A00010_2019-12-11.npy'
test_array = np.load(a)[:, :, 250, :]
test_array = np.array(test_array > 0, 'float32')
Functions.image_show(test_array[:, :, 1])
rim = get_rim(-test_array[:, :, 1])
Functions.image_show(rim)
rim_array, len_dict, loc_dict = rim_length_and_id(rim)
# energy_array, _, loc_dict, _ = get_connected_region_and_energy(test_array[:, :, 1], rim_array, len_dict)

weight, all_energy = calculate_balance_weights(test_array[:, :, 1], rim_enhance=1)
print(all_energy)
print(calculate_balance_weights(test_array[:, :, 1], return_stat=True))
Functions.image_show(weight[:, :])
