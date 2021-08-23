"""
see: get_sorted_connected_regions
input a 3D mask numpy array, output a dict, with key 1, 2, 3, ... (int), which conforms to the ranking of the volume of
the connected component. The value of the dict is lists of locations like {1: [(x1, y1, z1), (x2, y2, z2), ...], ...}
"""
import numpy as np
import Tool_Functions.Functions as Functions
import analysis.connected_region2d_and_scale_free_stat as rim_detect

np.set_printoptions(precision=10, suppress=True)
epsilon = 0.001


class DimensionError(Exception):
    def __init__(self, array):
        self.shape = np.shape(array)
        self.dimension = len(self.shape)

    def __str__(self):
        print("invalid dimension of", self.dimension, ", array has shape", self.shape)


def get_connected_regions(input_array, threshold=None, strict=False, start_id=None):
    """
    :param input_array: the mask array, with shape [x, y, z]
    :param threshold: the threshold of cast the mask array to binary
    :param strict: whether diagonal pixel is considered as adjacent.
    :param start_id: the connect region id
    :return: a dict, with key 1, 2, 3, ... (int), value is list of location: {1: [(x1, y1, z1), (x2, y2, z2), ...], ...}
             a dict, with key 1, 2, 3, ... (int), value is length(list of location)
             helper_array has shape [a, b, c, 2], first channel is the merge count, second for region id
             optional: start_id for next stage
    """
    if threshold is not None:
        input_array = np.array(input_array > threshold, 'float32')
    shape = np.shape(input_array)
    helper_array = np.zeros([shape[0], shape[1], shape[2], 2])
    # the last dim has two channels, the first is the key, the second is the volume
    helper_array[:, :, :, 0] = -input_array
    tracheae_points = np.where(helper_array[:, :, :, 0] < -epsilon)
    num_checking_points = len(tracheae_points[0])
    # print("we will check:", num_checking_points)
    id_volume_dict = {}
    id_loc_dict = {}

    if start_id is None:
        connected_id = 1
    else:
        connected_id = start_id

    for index in range(num_checking_points):
        pixel_location = (tracheae_points[0][index], tracheae_points[1][index], tracheae_points[2][index])
        if helper_array[pixel_location[0], pixel_location[1], pixel_location[2], 0] > epsilon:
            # this means this point has been allocated id and volume
            continue
        else:
            # this means this point has been allocated id and volume
            if strict:
                volume, locations = broadcast_connected_component(helper_array, pixel_location, connected_id)
            else:
                volume, locations = broadcast_connected_component_2(helper_array, pixel_location, connected_id)
            # now, the volume and id has been broadcast to this connected component.
            id_volume_dict[connected_id] = volume
            id_loc_dict[connected_id] = locations
            connected_id += 1  # the id is 1, 2, 3, ...

    if start_id is None:
        return id_volume_dict, id_loc_dict, helper_array
    else:
        return id_volume_dict, id_loc_dict, helper_array, connected_id


def get_connected_regions_light(input_flow, strict=False):
    """
    :param input_flow: the binary mask array, with shape [x, y, z], pid_id
    :param strict: whether diagonal pixel is considered as adjacent.
    :return: a dict, with key 1, 2, 3, ... (int), value is list of location: {1: [(x1, y1, z1), (x2, y2, z2), ...], ...}
    """
    input_array = input_flow[0]
    print("processing interval", input_flow[1])
    shape = np.shape(input_array)
    helper_array = np.zeros([shape[0], shape[1], shape[2], 2])
    # the last dim has two channels, the first is the key, the second is the volume
    helper_array[:, :, :, 0] = -input_array
    tracheae_points = np.where(helper_array[:, :, :, 0] < -epsilon)
    num_checking_points = len(tracheae_points[0])
    # print("we will check:", num_checking_points)
    id_loc_dict = {}
    connected_id = 1

    for index in range(num_checking_points):
        pixel_location = (tracheae_points[0][index], tracheae_points[1][index], tracheae_points[2][index])
        if helper_array[pixel_location[0], pixel_location[1], pixel_location[2], 0] > epsilon:
            # this means this point has been allocated id and volume
            continue
        else:
            # this means this point has been allocated id and volume
            if strict:
                volume, locations = broadcast_connected_component(helper_array, pixel_location, connected_id)
            else:
                volume, locations = broadcast_connected_component_2(helper_array, pixel_location, connected_id)
            # now, the volume and id has been broadcast to this connected component.
            id_loc_dict[connected_id] = locations
            connected_id += 1  # the id is 1, 2, 3, ...

    return id_loc_dict
    

def broadcast_connected_component(helper_array, initial_location, region_id):
    # helper_array has shape [a, b, c, 2]
    # initial_location is a tuple, (x, y, z)
    # return the volume of this connected_component (int) and the location list like [(389, 401), (389, 402), ..].
    volume = 0  # the volume of this connected component
    un_labeled_region = [initial_location, ]
    helper_array[initial_location[0], initial_location[1], initial_location[2], 1] = region_id
    region_locations = []
    while un_labeled_region:  # this mean un_labeled_region is not empty
        location = un_labeled_region.pop()

        region_locations.append(location)  # get the locations of the connected component
        volume += 1

        if helper_array[location[0] + 1, location[1], location[2], 0] < -epsilon:
            # whether the adjacent pixel is in the same connected_component
            if not helper_array[location[0] + 1, location[1], location[2], 1] == region_id:
                # this adjacent location is not visited
                un_labeled_region.append((location[0] + 1, location[1], location[2]))
                helper_array[location[0] + 1, location[1], location[2], 1] = region_id  # label this unlabeled pixel
        if helper_array[location[0] - 1, location[1], location[2], 0] < -epsilon:
            if not helper_array[location[0] - 1, location[1], location[2], 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1], location[2]))
                helper_array[location[0] - 1, location[1], location[2], 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2], 0] < -epsilon:
            if not helper_array[location[0], location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2]))
                helper_array[location[0], location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0], location[1] - 1, location[2], 0] < -epsilon:
            if not helper_array[location[0], location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2]))
                helper_array[location[0], location[1] - 1, location[2], 1] = region_id
        if helper_array[location[0], location[1], location[2] + 1, 0] < -epsilon:
            if not helper_array[location[0], location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] + 1))
                helper_array[location[0], location[1], location[2] + 1, 1] = region_id
        if helper_array[location[0], location[1], location[2] - 1, 0] < -epsilon:
            if not helper_array[location[0], location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] - 1))
                helper_array[location[0], location[1], location[2] - 1, 1] = region_id

    for location in region_locations:
        helper_array[location[0], location[1], location[2], 0] = volume
    # print('this component has id', region_id, 'volume', volume)
    return volume, region_locations


def broadcast_connected_component_2(helper_array, initial_location, region_id):
    # the difference is that here diagonal pixels are considered as adjacency.
    # helper_array has shape [a, b, c, 2]
    # initial_location is a tuple, (x, y, z)
    # return the volume of this connected_component (int) and the location list like [(389, 401), (389, 402), ..].
    volume = 0  # the volume of this connected component
    un_labeled_region = [initial_location, ]
    helper_array[initial_location[0], initial_location[1], initial_location[2], 1] = region_id
    region_locations = []
    while un_labeled_region:  # this mean un_labeled_region is not empty
        location = un_labeled_region.pop()

        region_locations.append(location)  # get the locations of the connected component
        volume += 1
        if not np.min(helper_array[location[0]-1:location[0]+2, location[1]-1:location[1]+2,
                      location[2]-1:location[2]+2]) < -epsilon:
            continue

        if helper_array[location[0] + 1, location[1], location[2], 0] < -epsilon:  # (1, 0, 0)
            # whether the adjacent pixel is in the same connected_component
            if not helper_array[location[0] + 1, location[1], location[2], 1] == region_id:
                # this adjacent location is not visited
                un_labeled_region.append((location[0] + 1, location[1], location[2]))
                helper_array[location[0] + 1, location[1], location[2], 1] = region_id  # label this unlabeled pixel
        if helper_array[location[0] - 1, location[1], location[2], 0] < -epsilon:
            if not helper_array[location[0] - 1, location[1], location[2], 1] == region_id:  # (-1, 0, 0)
                un_labeled_region.append((location[0] - 1, location[1], location[2]))
                helper_array[location[0] - 1, location[1], location[2], 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2], 0] < -epsilon:  # (0, 1, 0)
            if not helper_array[location[0], location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2]))
                helper_array[location[0], location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0], location[1] - 1, location[2], 0] < -epsilon:  # (0, -1, 0)
            if not helper_array[location[0], location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2]))
                helper_array[location[0], location[1] - 1, location[2], 1] = region_id
        if helper_array[location[0], location[1], location[2] + 1, 0] < -epsilon:  # (0, 0, 1)
            if not helper_array[location[0], location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] + 1))
                helper_array[location[0], location[1], location[2] + 1, 1] = region_id
        if helper_array[location[0], location[1], location[2] - 1, 0] < -epsilon:  # (0, 0, -1)
            if not helper_array[location[0], location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1], location[2] - 1))
                helper_array[location[0], location[1], location[2] - 1, 1] = region_id

        if helper_array[location[0] - 1, location[1] - 1, location[2], 0] < -epsilon:  # (-1, -1, 0)
            if not helper_array[location[0] - 1, location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] - 1, location[2]))
                helper_array[location[0] - 1, location[1] - 1, location[2], 1] = region_id
        if helper_array[location[0] - 1, location[1] + 1, location[2], 0] < -epsilon:  # (-1, 1, 0)
            if not helper_array[location[0] - 1, location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] + 1, location[2]))
                helper_array[location[0] - 1, location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0] + 1, location[1] + 1, location[2], 0] < -epsilon:  # (1, 1, 0)
            if not helper_array[location[0] + 1, location[1] + 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] + 1, location[2]))
                helper_array[location[0] + 1, location[1] + 1, location[2], 1] = region_id
        if helper_array[location[0] + 1, location[1] - 1, location[2], 0] < -epsilon:  # (1, -1, 0)
            if not helper_array[location[0] + 1, location[1] - 1, location[2], 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] - 1, location[2]))
                helper_array[location[0] + 1, location[1] - 1, location[2], 1] = region_id

        if helper_array[location[0] - 1, location[1] - 1, location[2] + 1, 0] < -epsilon:  # (-1, -1, 1)
            if not helper_array[location[0] - 1, location[1] - 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] - 1, location[2] + 1))
                helper_array[location[0] - 1, location[1] - 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] - 1, location[1] + 1, location[2] + 1, 0] < -epsilon:  # (-1, 1, 1)
            if not helper_array[location[0] - 1, location[1] + 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] + 1, location[2] + 1))
                helper_array[location[0] - 1, location[1] + 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] + 1, location[2] + 1, 0] < -epsilon:  # (1, 1, 1)
            if not helper_array[location[0] + 1, location[1] + 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] + 1, location[2] + 1))
                helper_array[location[0] + 1, location[1] + 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] - 1, location[2] + 1, 0] < -epsilon:  # (1, -1, 1)
            if not helper_array[location[0] + 1, location[1] - 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] - 1, location[2] + 1))
                helper_array[location[0] + 1, location[1] - 1, location[2] + 1, 1] = region_id

        if helper_array[location[0], location[1] - 1, location[2] + 1, 0] < -epsilon:  # (0, -1, 1)
            if not helper_array[location[0], location[1] - 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2] + 1))
                helper_array[location[0], location[1] - 1, location[2] + 1, 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2] + 1, 0] < -epsilon:  # (0, 1, 1)
            if not helper_array[location[0], location[1] + 1, location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2] + 1))
                helper_array[location[0], location[1] + 1, location[2] + 1, 1] = region_id
        if helper_array[location[0] + 1, location[1], location[2] + 1, 0] < -epsilon:  # (1, 0, 1)
            if not helper_array[location[0] + 1, location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1], location[2] + 1))
                helper_array[location[0] + 1, location[1], location[2] + 1, 1] = region_id
        if helper_array[location[0] - 1, location[1], location[2] + 1, 0] < -epsilon:  # (-1, 0, 1)
            if not helper_array[location[0] - 1, location[1], location[2] + 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1], location[2] + 1))
                helper_array[location[0] - 1, location[1], location[2] + 1, 1] = region_id

        if helper_array[location[0] - 1, location[1] - 1, location[2] - 1, 0] < -epsilon:  # (-1, -1, -1)
            if not helper_array[location[0] - 1, location[1] - 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] - 1, location[2] - 1))
                helper_array[location[0] - 1, location[1] - 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] - 1, location[1] + 1, location[2] - 1, 0] < -epsilon:  # (-1, 1, -1)
            if not helper_array[location[0] - 1, location[1] + 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1] + 1, location[2] - 1))
                helper_array[location[0] - 1, location[1] + 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] + 1, location[2] - 1, 0] < -epsilon:  # (1, 1, -1)
            if not helper_array[location[0] + 1, location[1] + 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] + 1, location[2] - 1))
                helper_array[location[0] + 1, location[1] + 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] + 1, location[1] - 1, location[2] - 1, 0] < -epsilon:  # (1, -1, -1)
            if not helper_array[location[0] + 1, location[1] - 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1] - 1, location[2] - 1))
                helper_array[location[0] + 1, location[1] - 1, location[2] - 1, 1] = region_id

        if helper_array[location[0], location[1] - 1, location[2] - 1, 0] < -epsilon:  # (0, -1, -1)
            if not helper_array[location[0], location[1] - 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] - 1, location[2] - 1))
                helper_array[location[0], location[1] - 1, location[2] - 1, 1] = region_id
        if helper_array[location[0], location[1] + 1, location[2] - 1, 0] < -epsilon:  # (0, 1, -1)
            if not helper_array[location[0], location[1] + 1, location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0], location[1] + 1, location[2] - 1))
                helper_array[location[0], location[1] + 1, location[2] - 1, 1] = region_id
        if helper_array[location[0] + 1, location[1], location[2] - 1, 0] < -epsilon:  # (1, 0, -1)
            if not helper_array[location[0] + 1, location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] + 1, location[1], location[2] - 1))
                helper_array[location[0] + 1, location[1], location[2] - 1, 1] = region_id
        if helper_array[location[0] - 1, location[1], location[2] - 1, 0] < -epsilon:  # (-1, 0, -1)
            if not helper_array[location[0] - 1, location[1], location[2] - 1, 1] == region_id:
                un_labeled_region.append((location[0] - 1, location[1], location[2] - 1))
                helper_array[location[0] - 1, location[1], location[2] - 1, 1] = region_id

    for location in region_locations:
        helper_array[location[0], location[1], location[2], 0] = volume
    # print('this component has id', region_id, 'volume', volume)
    return volume, region_locations


def sort_on_id_loc_dict(id_loc_dict, id_volume_dict=None):
    # refactor the key of the connected_components
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


def stat_on_connected_component(id_loc_dict, total_volume=None, show=True):  # total_volume is like the volume of lung
    keys_list = list(id_loc_dict.keys())
    if show:
        print("we have:", len(keys_list), "number of connected components")
    id_loc_dict_sorted, id_volume_dict_sorted = sort_on_id_loc_dict(id_loc_dict)
    if total_volume is None:
        if show:
            print("the volume of these components are:\n", id_volume_dict_sorted)
    else:
        if show:
            print("total_volume is:", total_volume)
        for key in keys_list:
            if show:
                print("component", key, "constitute:", id_volume_dict_sorted[key]/total_volume, "of total volume")
    return id_loc_dict_sorted


def get_sorted_connected_regions(input_array, threshold=0.5, strict=False, show=True):
    """
        :param input_array: the mask array, with shape [x, y, z]
        :param threshold: the threshold of cast the mask array to binary
        :param strict: whether diagonal pixel is considered as adjacent.
        :return id_loc_dict_sorted
        """
    # key start from 1: id_loc_dict_sorted[1] is the largest; threshold > 0.5 will be considered as positive, otherwise,
    # will be considered negative
    if len(np.shape(input_array)) == 3:
        id_volume_dict, id_loc_dict, helper_array = get_connected_regions(input_array, threshold=threshold,
                                                                          strict=strict)
        return stat_on_connected_component(id_loc_dict, show=show)
    elif len(np.shape(input_array)) == 2:
        shape = np.shape(input_array)
        temp_array = np.zeros((shape[0], shape[1], 3), 'float32')
        temp_array[:, :, 1] = input_array
        id_volume_dict, id_loc_dict, helper_array = get_connected_regions(temp_array, threshold=threshold,
                                                                          strict=strict)
        id_loc_dict_sorted = stat_on_connected_component(id_loc_dict, show=show)
        keys_list = list(id_loc_dict_sorted.keys())
        return_dict = {}
        for key in keys_list:
            return_dict[key] = list()
        for key in keys_list:
            for loc in id_loc_dict_sorted[key]:
                return_dict[key].append((loc[0], loc[1]))
        return return_dict
    else:
        raise DimensionError(input_array)


def connectedness_2d(loc_list, strict=False):
    """
    whether the loc_list forms a region that has the connectedness same to a circle?
    :param loc_list: a list of locations, like [(x1, y1), (x2, y2), ...]
    :param strict: if True, then diagonal pixel is considered as adjacent.
    :return:
    True if loc_list forms a region that has the connectedness same to a circle.
    False if otherwise, like their are more than one connected
    """
    x_min = 99999999999
    x_max = 0
    y_min = 99999999999
    y_max = 0
    for loc in loc_list:
        if loc[0] > x_max:
            x_max = loc[0]
        if loc[0] < x_min:
            x_min = loc[0]
        if loc[1] > y_max:
            y_max = loc[1]
        if loc[1] < y_min:
            y_min = loc[1]
    x_range = x_max - x_min
    y_range = y_max - y_min
    bounding_array = np.zeros((x_range + 6, y_range + 6), 'float32')
    for loc in loc_list:
        bounding_array[loc[0] - x_min + 3, loc[1] - y_min + 3] = 1
    # Functions.image_show(bounding_array)
    # we require there are only on connected component.
    assert len(list(get_sorted_connected_regions(bounding_array, strict=strict, show=False).keys())) == 1
    if not strict:
        rim_array = rim_detect.get_rim(bounding_array, outer=True)
        num_boundaries = len(list(get_sorted_connected_regions(rim_array, strict=strict, show=False).keys()))
        if num_boundaries == 1:
            return True
        else:
            print(num_boundaries)
            return False
    else:
        print("do not support strict adjacency")
        return None


if __name__ == '__main__':

    exit()
