import os
import numpy as np
import multiprocessing as mp
import math
import sys
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import Tool_Functions.Functions as Functions
import Analysis.connected_region2d_and_scale_free_stat as connected_region
import Analysis.connect_region_detect as connect_3d
np.set_printoptions(threshold=np.inf)
ibex = False
if not ibex:
    top_directory_rescaled_ct = '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_ct_follow_up/'
    # where the rescaled ct arrays saved: top_directory_rescaled_ct/patient_id/patient_id_time.npy
    top_directory_check_point = '/home/zhoul0a/Desktop/prognosis_project/check_points/'
    # where the checkpoints stored: top_directory_check_point/model_type/direction/best_model-direction.pth
    top_directory_masks = '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/'
    # where to save the predicted masks, which will form: top_directory_output/mask_type/patient_id/id_time_mask.npz
    top_directory_enhanced = '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_ct_enhanced_general_sampling/'
    # where to save the normalized ct without the air-way and blood-vessels and enhanced the lesion
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # use two V100 GPU
else:
    top_directory_rescaled_ct = '/ibex/scratch/projects/c2052/prognosis_project/'
    top_directory_check_point = '/ibex/scratch/projects/c2052/prognosis_project/'
    top_directory_output = '/ibex/scratch/projects/c2052/prognosis_project/'


def distance_l2(loc_1, loc_2):
    """
    :param loc_1: a tuple (x_1, y_1)
    :param loc_2: a tuple (x_2, y_2)
    :return: L2 distance between loc_1 and loc_2
    """
    x_difference = loc_1[0] - loc_2[0]
    y_difference = loc_1[1] - loc_2[1]
    return math.sqrt(x_difference*x_difference + y_difference*y_difference)


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


def get_max_diameter_one_region(loc_list, strict=False):
    """
    :param strict: if false, we believe the region is near round, then accelerate the speed by a great extent.
    :param loc_list: [(x_1, y_1), (x_2, y_2), ...]
    :return: a float, refer to the max L2 distance among these locations
    """
    max_diameter = 0
    num_locations = len(loc_list)
    if not strict:
        fist_loc = loc_list[0]
        for loc in loc_list[1::]:
            distance = distance_l2(fist_loc, loc)
            if distance > max_diameter:
                max_diameter = distance
        mid_loc = loc_list[int(num_locations / 3)]
        for loc in loc_list:
            distance = distance_l2(mid_loc, loc)
            if distance > max_diameter:
                max_diameter = distance
        mid_loc = loc_list[int(2 * num_locations / 3)]
        for loc in loc_list:
            distance = distance_l2(mid_loc, loc)
            if distance > max_diameter:
                max_diameter = distance
        return max_diameter
    else:
        for central_loc in loc_list[0:int(num_locations/2)]:
            for loc in loc_list:
                distance = distance_l2(central_loc, loc)
                if distance > max_diameter:
                    max_diameter = distance
        return max_diameter


def find_max_diameter_one_slice(id_loc_dict_rim):
    """
    :param id_loc_dict_rim: {connect_id: [(x_1, y_1), (x_2, y_2), ...]}
    :return: a dict, {connect_id: max L2 distance inside this region}
    """
    key_list = list(id_loc_dict_rim.keys())
    id_diameter_dict = {}
    for key in key_list:
        max_diameter = get_max_diameter_one_region(id_loc_dict_rim[key])
        id_diameter_dict[key] = max_diameter
    return id_diameter_dict


def find_max_diameter(rim_info_list):
    """
    :param rim_info_list: the return of: connected_region.abstract_connected_regions(z_axes_list, 'rim'):
    each element is [return_array, id_length_dict, id_loc_dict], return_array[:, :, 0] is the length map,
    return_array[:, :, 1] is the id map
    :return: a list of dict: [{connect_id: max_diameter}], each list element corresponding to rim_info_list.
    """
    return_list = []
    for rim_info in rim_info_list:
        id_loc_dict = rim_info[2]
        id_diameter_dict = find_max_diameter_one_slice(id_loc_dict)
        return_list.append(id_diameter_dict)
    return return_list


def extend_one_round_one_region(loc_rim, loc_region):
    """
    make the region bigger
    :param loc_rim: a set, record the locations of the rim points for this region, [(x_1, y_1), (x_2, y_2), ...]
    :param loc_region: a set, record ALL locations of this region, [(x_1, y_1), (x_2, y_2), ...]
    :return: the new loc_rim and the new loc_region (extended by one round)
    """
    new_loc_rim = set()
    for loc in loc_rim:
        new_loc_rim.add((loc[0] - 1, loc[1]))
        new_loc_rim.add((loc[0] + 1, loc[1]))
        new_loc_rim.add((loc[0], loc[1] - 1))
        new_loc_rim.add((loc[0], loc[1] + 1))
    # print("new_loc_rim", new_loc_rim)
    new_loc_rim = new_loc_rim - (new_loc_rim & loc_region)  # note for set, &, -, | are of same priority.
    # print("new", new_loc_rim)
    new_loc_region = new_loc_rim | loc_region
    return new_loc_rim, new_loc_region


def subtract_one_round_one_region(loc_rim, loc_region):
    """
    make the region smaller
    :param loc_rim: a set, record the locations of the rim points for this region, [(x_1, y_1), (x_2, y_2), ...]
    :param loc_region: a set, record ALL locations of this region, [(x_1, y_1), (x_2, y_2), ...]
    :return: the new loc_rim and the new loc_region (shrieked by one round)
    """
    new_loc_rim = set()
    for loc in loc_rim:
        new_loc_rim.add(loc)
        new_loc_rim.add((loc[0] - 1, loc[1]))
        new_loc_rim.add((loc[0] + 1, loc[1]))
        new_loc_rim.add((loc[0], loc[1] - 1))
        new_loc_rim.add((loc[0], loc[1] + 1))

    new_loc_rim = (new_loc_rim & loc_region) - loc_rim

    new_loc_region = loc_region - loc_rim

    return new_loc_rim, new_loc_region


def extend_one_slice(id_loc_dict_rim, id_loc_dict_region, extend_ratio=1.25, max_diameter=70):
    max_diameter_dict = find_max_diameter_one_slice(id_loc_dict_rim)
    key_list_2 = list(max_diameter_dict.keys())
    key_list = list(id_loc_dict_region.keys())
    assert len(key_list_2) >= len(key_list)

    for key in key_list:
        if max_diameter_dict[key] > max_diameter:
            max_diameter_dict[key] = max_diameter
    for region_id in key_list:
        loc_rim = set(id_loc_dict_rim[region_id])
        loc_region = set(id_loc_dict_region[region_id])
        if extend_ratio > 1:
            num_extend = round((extend_ratio - 1) * round(max_diameter_dict[region_id]) / 2)
            for layer in range(num_extend):
                new_loc_rim, new_loc_region = extend_one_round_one_region(loc_rim, loc_region)
                loc_rim = new_loc_rim
                loc_region = new_loc_region
                '''
                # observe how the rim is expand layer by layer
                image = np.zeros([512, 512], 'float32')
                print("loc_rim", loc_rim)
                for loc in loc_rim:
                    image[loc[0], loc[1]] = 1
                print(loc_rim)
                print("layer", layer)
                Functions.image_show(image)
                '''
            id_loc_dict_rim[region_id] = loc_rim
            id_loc_dict_region[region_id] = loc_region
        else:
            num_subtract = int((1 - extend_ratio) * int(max_diameter_dict[region_id]) / 2)
            for layer in range(num_subtract):
                new_loc_rim, new_loc_region = subtract_one_round_one_region(loc_rim, loc_region)
                loc_rim = new_loc_rim
                loc_region = new_loc_region
                '''
                # observe how the rim is expand layer by layer
                image = np.zeros([512, 512], 'float32')
                print("loc_rim", loc_rim)
                for loc in loc_rim:
                    image[loc[0], loc[1]] = 1
                print(loc_rim)
                print("layer", layer)
                Functions.image_show(image)
                '''
            id_loc_dict_rim[region_id] = loc_rim
            id_loc_dict_region[region_id] = loc_region

    return id_loc_dict_rim, id_loc_dict_region


def extend_tubes(input_mask, leave_connected_component=None, extend_ratio=1.25, max_diameter=50):
    """
    :param input_mask: binary numpy array with shape [x, y, z]
    :param leave_connected_component: how many 3D connected region we leave? if None means do not check 3D connectivity.
    :param extend_ratio: the mask perform good for bronchial but not very good for big air-way. extend the mask
    according to there diameter, which will avoid uncovered airway walls.
    :param max_diameter: if the diameter of the connected component is greater than the max_diameter, replace it by the
    max_diameter.
    :return: binary numpy array with shape [x, y, z] in 'float32'
    """
    if leave_connected_component is not None:
        id_loc_dict = connect_3d.get_sorted_connected_regions(input_mask, strict=False)
        refined_mask = np.zeros(np.shape(input_mask), 'float32')
        key = 1
        if len(list(id_loc_dict.keys())) < leave_connected_component:
            leave_connected_component = len(list(id_loc_dict.keys()))
        while key < leave_connected_component + 1:
            locations = id_loc_dict[key]
            for loc in locations:
                refined_mask[loc] = 1
            key += 1
        print("finally there are:", np.sum(refined_mask), "positive points")
    else:
        refined_mask = input_mask
    refined_mask = np.swapaxes(refined_mask, 0, 2)
    z_axes_list = list(refined_mask)

    extended_mask = np.zeros(np.shape(refined_mask), 'float32')
    rim_info_list, region_info_list = connected_region.abstract_connected_regions(z_axes_list, 'both')
    # the length of the rim_info_list is equal to that of region_info_list, like 512
    num_slices = len(z_axes_list)
    for slice_id in range(num_slices):
        _, extended_region_loc_dict = extend_one_slice(rim_info_list[slice_id][2], region_info_list[slice_id][2],
                                                       extend_ratio, max_diameter=max_diameter)
        key_list = list(extended_region_loc_dict.keys())
        for key in key_list:
            for loc in extended_region_loc_dict[key]:
                extended_mask[slice_id, loc[0], loc[1]] = 1
    extended_mask = np.swapaxes(extended_mask, 0, 2)
    return extended_mask


def remove_airway_and_blood_vessel_one_scan_upperlobe(patient_id, time, extend_ratio=1.1, max_diameter=50, show=True):
    rescaled_ct = np.load(top_directory_rescaled_ct + patient_id + '/' + patient_id + '_' + time + '.npy')
    lung_mask = np.load(top_directory_masks + 'lung_masks/' + patient_id + '/' + patient_id + '_' + time +
                        '_mask_refine.npz')['array']
    lung_mask_box = Functions.get_bounding_box(lung_mask)
    print("lung_mask_box:", lung_mask_box)
    lung_length_z = lung_mask_box[2][1] - lung_mask_box[2][0]
    superior_start = int(lung_mask_box[2][1] - lung_length_z * 0.32988803223955687)
    superior_end = lung_mask_box[2][1]
    air_way_merge_z_bounding_box = Functions.get_bounding_box(lung_mask[:, :, superior_start])
    upper_start = air_way_merge_z_bounding_box[0][0]
    upper_end = air_way_merge_z_bounding_box[0][1] - int((air_way_merge_z_bounding_box[0][1] -
                                                          air_way_merge_z_bounding_box[0][0])/2)
    print("lung length on z:", lung_length_z, "superior range:", superior_start, superior_end,
          "upper range:", upper_start, upper_end)
    upper_superior_mask = np.zeros(np.shape(lung_mask), 'float32')
    upper_superior_mask[upper_start: upper_end, :, superior_start: superior_end] = 1.0
    upper_superior_mask = upper_superior_mask * lung_mask

    refined_airway_mask = np.load(top_directory_masks + 'air_way_mask_stage_two/' + patient_id + '/' + patient_id + '_'
                                  + time + '_mask_refine.npz')['array']
    refined_blood_vessel_mask = np.load(top_directory_masks + 'blood_vessel_mask_stage_two/' + patient_id + '/' +
                                        patient_id + '_' + time + '_mask_refine.npz')['array']

    rescaled_ct = rescaled_ct * lung_mask
    visible_non_infection = np.array((refined_blood_vessel_mask + refined_airway_mask) * lung_mask > 0.5, 'float32')
    rescaled_ct_original = np.array(rescaled_ct)

    assert extend_ratio > 1
    print("extending air way")
    # visible_extended_outer = extend_tubes(visible_non_infection, None, extend_ratio + 0.1, max_diameter)
    visible_extended = extend_tubes(visible_non_infection, None, extend_ratio, max_diameter)
    context_mask = visible_extended - visible_non_infection
    context_mask = context_mask * upper_superior_mask
    num_context_points = np.sum(context_mask)
    context = context_mask * rescaled_ct_original + context_mask * 10
    context = np.reshape(context, (-1,))
    context = np.sort(context)
    total_points = len(context)
    percentile = 50
    threshold = context[total_points - int(num_context_points * (100 - percentile) / 100)] - 10
    print("the context is:", threshold, 'at', percentile)

    rescaled_ct[np.where(visible_non_infection >= 0.5)] = threshold
    rescaled_ct = rescaled_ct - threshold * lung_mask  # threshold is the value of lung parenchyma
    rescaled_ct_original = rescaled_ct_original - threshold * lung_mask
    save_array = np.zeros([512, 1024, 512], 'float32')
    save_array[:, 0:512, :] = rescaled_ct_original
    save_array[:, 512::, :] = rescaled_ct
    if show:
        image = np.zeros([512, 1024], 'float32')
        image[:, 0: 512] = rescaled_ct_original[:, :, 220]
        image[:, 512::] = rescaled_ct[:, :, 220]
        image = np.clip(image, 0, 0.2)
        Functions.image_save(image, '/home/zhoul0a/Desktop/prognosis_project/visualize/remove_visible_upperlobe/'+patient_id + '_'+time+'_compare.png', high_resolution=True,
                             gray=True)
        superior_middle = int((superior_end + superior_start) / 2)
        image = np.zeros([512, 1024, 3], 'float32')
        image[:, 0: 512, 0] = rescaled_ct_original[:, :, superior_middle]
        image[:, 0: 512, 1] = rescaled_ct_original[:, :, superior_middle]
        image[:, 0: 512, 2] = rescaled_ct_original[:, :, superior_middle]
        image[:, 512::, 0] = rescaled_ct_original[:, :, superior_middle] + context_mask[:, :, superior_middle]
        image[:, 512::, 1] = rescaled_ct_original[:, :, superior_middle] - context_mask[:, :, superior_middle]
        image[:, 512::, 2] = rescaled_ct_original[:, :, superior_middle] - context_mask[:, :, superior_middle]
        image = np.clip(image, 0, 0.2)
        Functions.image_save(image, '/home/zhoul0a/Desktop/prognosis_project/visualize/remove_visible_upperlobe/' + patient_id + '_' + time+ '_airway.png', high_resolution=True,
                             gray=False)
    Functions.save_np_array(top_directory_enhanced, patient_id + '_' + time, save_array)


def remove_airway_and_blood_vessel_one_scan_general_sampling(patient_id, time, extend_ratio=1.1, max_diameter=50, show=True):
    rescaled_ct = np.load(top_directory_rescaled_ct + patient_id + '/' + patient_id + '_' + time + '.npy')
    lung_mask = np.load(top_directory_masks + 'lung_masks/' + patient_id + '/' + patient_id + '_' + time +
                        '_mask_refine.npz')['array']

    refined_airway_mask = np.load(top_directory_masks + 'air_way_mask_stage_two/' + patient_id + '/' + patient_id + '_'
                                  + time + '_mask_refine.npz')['array']
    refined_blood_vessel_mask = np.load(top_directory_masks + 'blood_vessel_mask_stage_two/' + patient_id + '/' +
                                        patient_id + '_' + time + '_mask_refine.npz')['array']

    rescaled_ct = rescaled_ct * lung_mask
    visible_non_infection = np.array((refined_blood_vessel_mask + refined_airway_mask) * lung_mask > 0.5, 'float32')
    rescaled_ct_original = np.array(rescaled_ct)

    assert extend_ratio > 1
    print("extending air way")
    # visible_extended_outer = extend_tubes(visible_non_infection, None, extend_ratio + 0.1, max_diameter)
    visible_extended = extend_tubes(visible_non_infection, None, extend_ratio, max_diameter)
    context_mask = visible_extended - visible_non_infection
    num_context_points = np.sum(context_mask)
    context = context_mask * rescaled_ct_original + context_mask * 10
    context = np.reshape(context, (-1,))
    context = np.sort(context)
    total_points = len(context)
    percentile = 50
    threshold = context[total_points - int(num_context_points * (100 - percentile) / 100)] - 10
    print("the context is:", threshold, 'at', percentile)

    rescaled_ct[np.where(visible_non_infection >= 0.5)] = threshold
    rescaled_ct = rescaled_ct - threshold * lung_mask  # threshold is the value of lung parenchyma
    rescaled_ct_original = rescaled_ct_original - threshold * lung_mask
    save_array = np.zeros([512, 1024, 512], 'float32')
    save_array[:, 0:512, :] = rescaled_ct_original
    save_array[:, 512::, :] = rescaled_ct
    if show:
        image = np.zeros([512, 1024], 'float32')
        image[:, 0: 512] = rescaled_ct_original[:, :, 280]
        image[:, 512::] = rescaled_ct[:, :, 280]
        image = np.clip(image, 0, 0.2)
        Functions.image_save(image, '/home/zhoul0a/Desktop/prognosis_project/visualize/remove_visible_general_sampling/'+patient_id + '_'+time+'_compare.png', high_resolution=True,
                             gray=True)
        image = np.zeros([512, 1024, 3], 'float32')
        image[:, 0: 512, 0] = rescaled_ct_original[:, :, 280]
        image[:, 0: 512, 1] = rescaled_ct_original[:, :, 280]
        image[:, 0: 512, 2] = rescaled_ct_original[:, :, 280]
        image[:, 512::, 0] = rescaled_ct_original[:, :, 280] + context_mask[:, :, 280]
        image[:, 512::, 1] = rescaled_ct_original[:, :, 280] - context_mask[:, :, 280]
        image[:, 512::, 2] = rescaled_ct_original[:, :, 280] - context_mask[:, :, 280]
        image = np.clip(image, 0, 0.2)
        Functions.image_save(image, '/home/zhoul0a/Desktop/prognosis_project/visualize/remove_visible_general_sampling/' + patient_id + '_' + time+ '_airway.png', high_resolution=True,
                             gray=False)
    Functions.save_np_array(top_directory_enhanced, patient_id + '_' + time, save_array)


if __name__ =='__main__':
    exit()
    import Tool_Functions.id_time_generator as id_time_generator

    id_time_list = id_time_generator.return_all_tuples_for_rescaled_ct(
        '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_ct_follow_up/')
    print("there are", len(id_time_list), 'scans')
    for data in id_time_list:
        print(data)
        remove_airway_and_blood_vessel_one_scan_general_sampling(data[0], data[1], extend_ratio=1.1)
        remove_airway_and_blood_vessel_one_scan_upperlobe(data[0], data[1], extend_ratio=1.1)

