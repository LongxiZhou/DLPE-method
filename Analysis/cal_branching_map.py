import Analysis.connected_region2d_and_scale_free_stat as connectivity
import Tool_Functions.Functions as Functions
import random
import Format_convert.spatial_normalize as normalize
import numpy as np
import os

# use function: "calculate_branching_map"

interval = 3
# if your computer is slow, increase the interval will accelerate interval^3 times
# if interval = 2, one scan need ~6 cpu hours, and the program is well-paralleled.
leave_cpu = 2

show_details = True  # if your want to see details during processing, let it true


def relative_branching_level_semantic(blood_vessel_mask):
    """
    estimate the radius of the blood_vessel_points.
    :param blood_vessel_mask: shaped [512, 512, 512], 0, 1 mask
    :return: freq [512, 512, 512] float32 array, 0 for not blood_vessel, and > 0 for inside blood_vessel, which is the
    estimated radius.
    """
    shape = np.shape(blood_vessel_mask)
    assert shape == (512, 512, 512)

    # x direction:
    input_list = []
    for x in range(512):
        input_list.append(blood_vessel_mask[x, :, :])
    output_list = connectivity.abstract_connected_regions(input_list, aspect='area')
    relative_branching_x = np.zeros(shape, 'float32')
    for x in range(512):
        relative_branching_x[x, :, :] = output_list[x][0][:, :, 0]

    # y direction
    input_list = []
    for y in range(512):
        input_list.append(blood_vessel_mask[:, y, :])
    output_list = connectivity.abstract_connected_regions(input_list, aspect='area')
    relative_branching_y = np.zeros(shape, 'float32')
    for y in range(512):
        relative_branching_y[:, y, :] = output_list[y][0][:, :, 0]

    # z direction
    input_list = []
    for z in range(512):
        input_list.append(blood_vessel_mask[:, :, z])
    output_list = connectivity.abstract_connected_regions(input_list, aspect='area')
    relative_branching_z = np.zeros(shape, 'float32')
    for z in range(512):
        relative_branching_z[:, :, z] = output_list[z][0][:, :, 0]

    temp_array = np.array([relative_branching_x, relative_branching_y, relative_branching_z])
    print(np.shape(temp_array))

    return np.sqrt(np.min(temp_array, axis=0) / 3.1415926)


def get_relative_branching_map(lung_mask, blood_vessel_mask, show=True):
    blood_vessel_surface = get_surface_of_mask(blood_vessel_mask) * lung_mask
    blood_vessel_branching_level = relative_branching_level_semantic(blood_vessel_mask)
    relative_branching_map = np.zeros([512, 512, 512], 'float32')
    cube_generator = np.zeros([592, 592, 592, 3], 'float32')
    cube_generator[30: 542, 30: 542, 30: 542, 0] = blood_vessel_surface
    cube_generator[30: 542, 30: 542, 30: 542, 1] = blood_vessel_branching_level
    cube_generator[30: 542, 30: 542, 30: 542, 2] = lung_mask
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = \
        Functions.get_bounding_box(cube_generator[:, :, :, 2])
    if show:
        print("bounding_box:", (x_min, x_max), (y_min, y_max), (z_min, z_max))
    cube_list = []
    x = x_min - 30
    while x <= x_max - 30:
        if show:
            print("processing x:", x)
        y = y_min - 30
        while y <= y_max - 30:
            if show:
                print("processing y:", y)
            z = z_min - 30
            while z <= z_max - 30:
                if show:
                    print("processing z:", z)
                if np.sum(cube_generator[x + 30: x + 60, y + 30: y + 60, z + 30: z + 60, 2]) > 0:
                    if np.sum(cube_generator[x: x + 90, y: y + 90, z: z + 90, 0]) > 0:
                        cube_list.append(cube_generator[x: x + 90, y: y + 90, z: z + 90, 0: 2])
                z += 30
            y += 30
        x += 30
    solution_list = Functions.func_parallel(get_relative_branching_level_one_cube, cube_list, leave_cpu_num=leave_cpu)

    cube_id = 0
    x = x_min - 30
    while x <= x_max - 30:
        if show:
            print("processing x:", x)
        y = y_min - 30
        while y <= y_max - 30:
            if show:
                print("processing y:", y)
            z = z_min - 30
            while z <= z_max - 30:
                if show:
                    print("processing z:", z)
                if np.sum(cube_generator[x + 30: x + 60, y + 30: y + 60, z + 30: z + 60, 2]) > 0:
                    if np.sum(cube_generator[x: x + 90, y: y + 90, z: z + 90, 0]) > 0:
                        # notice that relative_branching has shape [512, 512, 512]
                        shape = np.shape(relative_branching_map[x: x + 30, y: y + 30, z: z + 30])
                        relative_branching_map[x: x + 30, y: y + 30, z: z + 30] = \
                            solution_list[cube_id][0: shape[0], 0: shape[1], 0: shape[2]]
                        cube_id += 1
                z += 30
            y += 30
        x += 30

    relative_branching_map = relative_branching_map * lung_mask
    return relative_branching_map


def get_surface_of_mask(blood_vessel_mask):
    # return the surface mask
    print("originally there are", np.sum(blood_vessel_mask))
    surface_array = connectivity.get_rim(blood_vessel_mask, outer=False)
    print("surface pixel number is", np.sum(surface_array))
    return surface_array


def get_nearest_mask_location(inputs):  # return (x, y, z), which is the nearest mask point
    location_parenchyma = inputs[0]  # freq tuple of location
    adjacent_mask_locations = inputs[1]  # freq list of mask locations
    min_distance_square = 10000000
    min_location = None
    for mask_location in adjacent_mask_locations:
        distance_square = 0
        distance_square += (location_parenchyma[0] - mask_location[0]) * (location_parenchyma[0] - mask_location[0])
        distance_square += (location_parenchyma[1] - mask_location[1]) * (location_parenchyma[1] - mask_location[1])
        distance_square += (location_parenchyma[2] - mask_location[2]) * (location_parenchyma[2] - mask_location[2])
        if distance_square < min_distance_square:
            min_distance_square = distance_square
            min_location = mask_location
    return min_location


def get_relative_branching_level_one_cube(mask_cube_and_level):
    # input freq [l, l, l, 2] cube, with [l, l, l, 0] is the mask_surface and [l, l, l, 1] is the estimated radius.
    # return freq [l/3, l/3, l/3] cube, each pixel value is the nearest radius of the mask. l % 3 should be 0!
    global interval
    shape_cube = np.shape(mask_cube_and_level)
    assert len(shape_cube) == 4
    assert shape_cube[0] % 3 == shape_cube[1] % 3 == shape_cube[2] % 3 == 0
    length = shape_cube[0]
    locations = np.where(mask_cube_and_level[:, :, :, 0] > 0.5)
    loc_surface = list(zip(list(locations[0]), list(locations[1]), list(locations[2])))  # list of the mask surface loc
    random.shuffle(loc_surface)
    while len(loc_surface) > 2000:
        loc_surface = loc_surface[0:int(0.8 * len(loc_surface))]

    return_array_extended = np.zeros(shape_cube[0: 3], 'float32')
    start = int(length / 3)
    return_array = np.zeros([start, start, start], 'float32')

    return_array_extended[start: 2 * start: interval, start: 2 * start: interval, start: 2 * start: interval] = 1

    locations = np.where(return_array_extended[:, :, :] > 0.5)
    loc_waiting = list(zip(list(locations[0]), list(locations[1]), list(locations[2])))
    # list of the loc waiting to calculate the relative branching

    return_list = []
    for loc in loc_waiting:
        return_list.append(get_nearest_mask_location((loc, loc_surface)))

    number_calculated = len(return_list)

    # print("num_waiting:", number_calculated, "num_surface:", len(loc_surface))
    for i in range(number_calculated):
        loc = loc_waiting[i]
        loc_nearest = return_list[i]
        return_array[loc[0] - start, loc[1] - start, loc[2] - start] = \
            mask_cube_and_level[loc_nearest[0], loc_nearest[1], loc_nearest[2], 1]
    return return_array


def convert_to_final_branching_map(scattered_branching_array):
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = Functions.get_bounding_box(scattered_branching_array)
    temp_array = scattered_branching_array[x_min: x_max+1: interval, y_min: y_max+1: interval, z_min: z_max+1: interval]
    return_array = np.zeros(np.shape(scattered_branching_array), 'float32')
    rescaled = normalize.rescale_to_standard(temp_array, [interval, interval, interval], [1, 1, 1],
                                             [x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1])
    return_array[x_min: x_max + 1, y_min: y_max + 1, z_min: z_max + 1] = rescaled
    return return_array


def calculate_branching_map(lung_mask, blood_vessel_mask, show=False):

    scattered_map = get_relative_branching_map(lung_mask, blood_vessel_mask, show)
    final_array = convert_to_final_branching_map(scattered_map) * lung_mask
    # there are very little pixels on the surface of the lungs did not calculate
    locations = np.where((lung_mask - np.array(final_array > 0) > 0))
    loc_did_not_cal = list(zip(list(locations[0]), list(locations[1]), list(locations[2])))
    for loc in loc_did_not_cal:
        final_array[loc[0], loc[1], loc[2]] = \
            np.min(final_array[loc[0] - 2: loc[0] + 3, loc[1] - 2: loc[1] + 3, loc[2] - 2: loc[2] + 3])
    final_array[np.where(final_array > 0)] = np.log(final_array[np.where(final_array > 0)])
    # final array is ln(a_0) + level * ln(alpha)

    min_value = -0.5  # branching at level 10; -0.5 = ln(a_0) + 10 * ln(alpha)
    max_value = np.max(final_array)  # branching at level 1; max_value = ln(a_0) + ln(alpha)

    log_alpha = (min_value - max_value) / 9
    log_a_0 = -0.5 - 10 * log_alpha

    print('A_0', np.exp(log_a_0), 'alpha', np.exp(log_alpha))
    branching_level_array = (final_array - log_a_0) / log_alpha * lung_mask * (1 - blood_vessel_mask)

    branching_level_array = np.clip(branching_level_array, 0, 12)
    # reliable blood vessel segmentation can reach to level 12

    return branching_level_array


if __name__ == '__main__':
    array_list = os.listdir('/home/zhoul0a/Desktop/COVID-19 delta/rescaled_arrays')
    total_number = int(len(array_list))
    processed = 0

    for array_name in array_list:
        print('processing:', array_name, total_number - processed, 'left\n')

        if array_name.split('_')[0] in ['tang-hai-feng']:
            print("wrong")
            continue

        if os.path.exists('/home/zhoul0a/Desktop/COVID-19 delta/branching_array/' + array_name[:-4] + '.npz'):
            print("processed")
            processed += 1
            continue

        lung = np.load('/home/zhoul0a/Desktop/COVID-19 delta/masks/lung_mask/' + array_name[:-4] + '.npz')['array']
        blood = np.load('/home/zhoul0a/Desktop/COVID-19 delta/masks/blood_vessel/' + array_name[:-4] + '.npz')['array']

        branching_map = calculate_branching_map(lung, blood)

        Functions.save_np_array('/home/zhoul0a/Desktop/COVID-19 delta/branching_array/', array_name[:-4],
                                branching_map, compress=True)
        processed += 1
