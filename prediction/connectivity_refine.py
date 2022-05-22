import Analysis.connect_region_detect as connect_region_detect
import os
import sys
import numpy as np

sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import Tool_Functions.Functions as Functions


def refine_masks_all(mask_dict, save_dict, lung_mask_dict, component_num=7):
    """
    :param lung_mask_dict: stores the masks of lungs, in mask_dict/patient_id/id_time_mask.npz, None means do not use
    lung mask.
    :param mask_dict: stores the masks being refining, in mask_dict/patient_id/id_time_mask.npz
    :param save_dict: where to save, in save_dict/patient_id/id_time_mask_refine.npz
    :param component_num:  maximum number of connected component to leave
    :return: None
    """
    print("mask_dict:", mask_dict)

    patient_id_list = os.listdir(mask_dict)
    num_patients = len(patient_id_list)
    print("total:", num_patients, "patients")
    for patient in patient_id_list:
        print("processing:", patient, num_patients, "left")
        time_dict = os.path.join(mask_dict, patient)
        if lung_mask_dict is not None:
            time_dict_lung = os.path.join(lung_mask_dict, patient)
        else:
            time_dict_lung = None
        time_list = os.listdir(time_dict)
        num_scans = len(time_list)
        print("there are:", num_scans, "scans for patient", patient)
        for time in time_list:
            print("processing:", patient, 'time:', time, num_scans, "left")

            save_path = os.path.join(save_dict, patient) + '/' + time[:-4] + '_refine.npz'
            if os.path.exists(save_path):
                print("processed")
                continue
            if time[-10] == 'r':
                print("processed")
                continue

            mask_path = os.path.join(time_dict, time)
            if time_dict_lung is not None:
                lung_mask_path = os.path.join(time_dict_lung, time)
            else:
                lung_mask_path = None
            assert mask_path[-1] == 'z'
            prediction = np.load(mask_path)['array']
            if lung_mask_path is not None:
                lung_mask = np.load(lung_mask_path)['array']
                prediction = prediction * lung_mask
            else:
                prediction[0, :, :] = 0
                prediction[-1, :, :] = 0
                prediction[:, 0, :] = 0
                prediction[:, -1, :] = 0
                prediction[:, :, 0] = 0
                prediction[:, :, -1] = 0

            print("originally there are:", np.sum(prediction), 'positive points')
            final_prediction = np.zeros(np.shape(prediction), 'float32')
            id_loc_dict = connect_region_detect.get_sorted_connected_regions(prediction)
            length_max = len(id_loc_dict[1])
            print("the max component:", length_max)
            key = 1
            if len(list(id_loc_dict.keys())) < component_num:
                component_num = len(list(id_loc_dict.keys()))
            while key < component_num + 1:
                locations = id_loc_dict[key]
                if len(locations) * 1000 < length_max:
                    print("too small components, finally use:", key - 1, "num of components")
                    break
                for loc in locations:
                    final_prediction[loc] = 1
                key += 1
            print("finally there are:", np.sum(final_prediction), "positive points")

            Functions.save_np_array(os.path.join(save_dict, patient) + '/', time[:-4] + '_refine.npz', final_prediction,
                                    True)
            num_scans -= 1
        num_patients -= 1


def refine_masks_all_parallel(mask_dict, save_dict, lung_mask_dict, component_num=7, parallel_count=24):
    """
    mask_dict and lung_mask_dict both in:
    mask_dict/patient_id/id_time_mask.npz
    save as:
    save_dict/patient_id/id_time_mask_refine.npz
    :param lung_mask_dict: stores the masks of lungs, in mask_dict/patient_id/id_time_mask.npz, None means do not use
    lung mask.
    :param mask_dict: stores the masks being refining, in mask_dict/patient_id/id_time_mask.npz
    :param save_dict: where to save, in save_dict/patient_id/id_time_mask_refine.npz
    :param component_num:  maximum number of connected component to leave
    :param parallel_count: how many programs run simultaneously
    :return: None
    """
    input_list = []
    for i in range(parallel_count):
        input_list.append([mask_dict, save_dict, lung_mask_dict, component_num, parallel_count, i])

    Functions.func_parallel(refine_masks_all_one_thread, input_list, 8)


def refine_masks_all_parallel_2(mask_dict, save_dict, lung_mask_dict, component_num=7, parallel_count=24):
    """
    mask_dict and lung_mask_dict both in:
    mask_dict/id_time_mask.npz
    save as:
    save_dict/id_time_mask_refine.npz
    :param lung_mask_dict: stores the masks of lungs, in mask_dict/id_time_mask.npz, None means do not use
    lung mask.
    :param mask_dict: stores the masks being refining, in mask_dict/id_time_mask.npz
    :param save_dict: where to save, in save_dict/id_time_mask_refine.npz
    :param component_num: maximum number of connected component to leave
    :param parallel_count: how many programs run simultaneously
    :return: None
    """
    input_list = []
    for i in range(parallel_count):
        input_list.append([mask_dict, save_dict, lung_mask_dict, component_num, parallel_count, i])

    Functions.func_parallel(refine_masks_all_one_thread_2, input_list, 8)


def refine_masks_all_one_thread(inputs):
    """
    mask_dict and lung_mask_dict both in:
    mask_dict/patient_id/id_time_mask.npz
    save as:
    save_dict/patient_id/id_time_mask_refine.npz
    :param inputs: tuple of: (mask_dict, save_dict, lung_mask_dict, component_num, parallel_count, thread_id)
    :return: None
    """
    mask_dict, save_dict, lung_mask_dict, component_num, parallel_count, thread_id = inputs
    scan_id = 0
    patient_id_list = os.listdir(mask_dict)
    num_patients = len(patient_id_list)

    for patient in patient_id_list:
        time_dict = os.path.join(mask_dict, patient)
        if lung_mask_dict is not None:
            time_dict_lung = os.path.join(lung_mask_dict, patient)
        else:
            time_dict_lung = None
        time_list = os.listdir(time_dict)
        num_scans = len(time_list)
        for time in time_list:
            if scan_id % parallel_count == thread_id:
                print("processing:", patient, 'time:', time, num_scans, "left")
            else:
                scan_id += 1
                continue
            save_path = os.path.join(save_dict, patient) + '/' + time[:-4] + '_refine.npz'
            if os.path.exists(save_path):
                print("processed")
                scan_id += 1
                continue
            if time[-10] == 'r':
                print("processed")
                scan_id += 1
                continue

            mask_path = os.path.join(time_dict, time)
            if time_dict_lung is not None:
                lung_mask_path = os.path.join(time_dict_lung, time)
            else:
                lung_mask_path = None
            assert mask_path[-1] == 'z'
            prediction = np.load(mask_path)['array']
            if lung_mask_path is not None:
                lung_mask = np.load(lung_mask_path)['array']
                prediction = prediction * lung_mask
            else:
                prediction[0, :, :] = 0
                prediction[-1, :, :] = 0
                prediction[:, 0, :] = 0
                prediction[:, -1, :] = 0
                prediction[:, :, 0] = 0
                prediction[:, :, -1] = 0

            print("originally there are:", np.sum(prediction), 'positive points')
            final_prediction = np.zeros(np.shape(prediction), 'float32')
            id_loc_dict = connect_region_detect.get_sorted_connected_regions(prediction)
            length_max = len(id_loc_dict[1])
            print("the max component:", length_max)
            key = 1
            if len(list(id_loc_dict.keys())) < component_num:
                component_num = len(list(id_loc_dict.keys()))
            while key < component_num + 1:
                locations = id_loc_dict[key]
                if len(locations) * 1000 < length_max:
                    print("too small components, finally use:", key - 1, "num of components")
                    break
                for loc in locations:
                    final_prediction[loc] = 1
                key += 1
            print("finally there are:", np.sum(final_prediction), "positive points")

            Functions.save_np_array(os.path.join(save_dict, patient) + '/', time[:-4] + '_refine.npz', final_prediction,
                                    True)
            num_scans -= 1
            scan_id += 1
        num_patients -= 1


def refine_masks_all_one_thread_2(inputs):
    """
    mask_dict and lung_mask_dict both in:
    mask_dict/id_time_mask.npz
    save as:
    save_dict/id_time_mask_refine.npz
    :param inputs: tuple of: (mask_dict, save_dict, lung_mask_dict, component_num, parallel_count, thread_id)
    :return: None
    """
    mask_dict, save_dict, lung_mask_dict, component_num, parallel_count, thread_id = inputs
    scan_id = 0
    mask_name_list = os.listdir(mask_dict)
    num_masks = len(mask_name_list)
    if not save_dict[-1] == '/':
        save_dict = save_dict + '/'

    for mask_name in mask_name_list:
        if lung_mask_dict is not None:  # use lung mask means further refined = refined * lung_mask
            lung_mask_path = os.path.join(lung_mask_dict, mask_name)
        else:
            lung_mask_path = None
        if scan_id % parallel_count == thread_id:
            print("processing:", mask_name, "scan id:", scan_id, "total:", num_masks)
        else:
            scan_id += 1
            continue
        save_path = os.path.join(save_dict, mask_name[:-4] + '_refine.npz')
        if os.path.exists(save_path):
            print("processed")
            scan_id += 1
            continue
        if mask_name[-10] == 'r':
            print("processed")
            scan_id += 1
            continue

        mask_path = os.path.join(mask_dict, mask_name)
        assert mask_path[-1] == 'z'
        prediction = np.load(mask_path)['array']
        if lung_mask_path is not None:
            lung_mask = np.load(lung_mask_path)['array']
            prediction = prediction * lung_mask
        else:
            prediction[0, :, :] = 0
            prediction[-1, :, :] = 0
            prediction[:, 0, :] = 0
            prediction[:, -1, :] = 0
            prediction[:, :, 0] = 0
            prediction[:, :, -1] = 0

        print("originally there are:", np.sum(prediction), 'positive points')
        final_prediction = np.zeros(np.shape(prediction), 'float32')
        id_loc_dict = connect_region_detect.get_sorted_connected_regions(prediction)
        length_max = len(id_loc_dict[1])
        print("the max component:", length_max)
        key = 1
        if len(list(id_loc_dict.keys())) < component_num:
            component_num = len(list(id_loc_dict.keys()))
        while key < component_num + 1:
            locations = id_loc_dict[key]
            if len(locations) * 1000 < length_max:
                print("too small components, finally use:", key - 1, "num of components")
                break
            for loc in locations:
                final_prediction[loc] = 1
            key += 1
        print("finally there are:", np.sum(final_prediction), "positive points")

        Functions.save_np_array(save_dict, mask_name[:-4] + '_refine.npz', final_prediction,
                                True)
        scan_id += 1


def refine_mask(mask, restrictive_mask=None, connected_num=7, lowest_ratio=0.001):
    """
    :param lowest_ratio: if a component volume is less than the lowest_ratio * max_volume_component, neglect it
    :param mask: the mask we need to refine
    :param restrictive_mask: if not None, means we use the mask = restrictive_mask * lung_mask to pre-refine data
    :param connected_num:  maximum number of connected component to leave
    :return: the refined mask
    """
    if restrictive_mask is not None:
        mask = restrictive_mask * mask
    else:
        mask[0, :, :] = 0
        mask[-1, :, :] = 0
        mask[:, 0, :] = 0
        mask[:, -1, :] = 0
        mask[:, :, 0] = 0
        mask[:, :, -1] = 0

    print("originally there are:", np.sum(mask), 'positive points')
    refined_mask = np.zeros(np.shape(mask), 'float32')
    id_loc_dict = connect_region_detect.get_sorted_connected_regions(mask)
    length_max = len(id_loc_dict[1])
    print("the max component:", length_max)
    key = 1
    if len(list(id_loc_dict.keys())) < connected_num:
        connected_num = len(list(id_loc_dict.keys()))
    while key < connected_num + 1:
        locations = id_loc_dict[key]
        if len(locations) < length_max * lowest_ratio:
            print("too small components, finally use:", key - 1, "num of components")
            break
        for loc in locations:
            refined_mask[loc] = 1
        key += 1
    print("finally there are:", np.sum(refined_mask), "positive points")

    return refined_mask


if __name__ == '__main__':

    refine_masks_all_parallel('/home/zhoul0a/Desktop/normal_people/rescaled_masks/air_way_mask_stage_two',
                              '/home/zhoul0a/Desktop/normal_people/rescaled_masks_refined/air_way_mask_stage_two',
                              '/home/zhoul0a/Desktop/normal_people/rescaled_masks/lung_masks',
                              component_num=7, parallel_count=40)
    exit()
    refine_masks_all('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/lung_masks/',
                     '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/lung_masks/',
                     '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/lung_masks/',
                     component_num=2)
    exit()
    refine_masks_all(
        '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/air_way_mask_stage_two/',
        '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/air_way_mask_stage_two/',
        '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/lung_masks/')
    refine_masks_all(
        '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/blood_vessel_mask_stage_two/',
        '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/blood_vessel_mask_stage_two/',
        '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/lung_masks/')
