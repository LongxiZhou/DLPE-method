import read_in_CT
import process_mha
import Functions
import os
import numpy as np
import warnings
import cv2

# top_dic = "/ibex/scratch/projects/c2052/COVID-19"
# patient_id_list = Functions.get_patient_id()

# for patient in patient_id_list:
#     read_in_CT.get_info(patient, show=True)


def rescale_to_standard(array, resolution, target_resolution=(334/512, 334/512, 1), target_shape=(512, 512, 512)):
    # pad and rescale the array to the same resolution and shape for further processing.
    # input: array must has shape (x, y, z) and resolution is a list or tuple with three elements

    original_shape = np.shape(array)
    target_volume = (target_resolution[0]*target_shape[0], target_resolution[1]*target_shape[1], target_resolution[2]*target_shape[2])
    shape_of_target_volume = (int(target_volume[0]/resolution[0]), int(target_volume[1]/resolution[1]), int(target_volume[2]/resolution[2]))

    if original_shape[2] * resolution[2] > target_volume[2]:
        warnings.warn('z-axis is longer than expectation. Make sure lung is near the center of z-axis.', SyntaxWarning)
        array = array[:, :, 100::]
        original_shape = np.shape(array)

    x = max(shape_of_target_volume[0], original_shape[0]) + 2
    y = max(shape_of_target_volume[1], original_shape[1]) + 2
    z = max(shape_of_target_volume[2], original_shape[2]) + 2

    x_start = int(x/2)-int(original_shape[0]/2)
    x_end = x_start + original_shape[0]
    y_start = int(y/2)-int(original_shape[1]/2)
    y_end = y_start + original_shape[1]
    z_start = int(z / 2) - int(original_shape[2] / 2)
    z_end = z_start + original_shape[2]

    array_intermediate = np.zeros((x, y, z), 'float32')
    array_intermediate[x_start:x_end, y_start:y_end, z_start:z_end] = array

    x_start = int(x / 2) - int(shape_of_target_volume[0] / 2)
    x_end = x_start + shape_of_target_volume[0]
    y_start = int(y / 2) - int(shape_of_target_volume[1] / 2)
    y_end = y_start + shape_of_target_volume[1]
    z_start = int(z / 2) - int(shape_of_target_volume[2] / 2)
    z_end = z_start + shape_of_target_volume[2]

    array_intermediate = array_intermediate[x_start:x_end, y_start:y_end, z_start:z_end]  # Now the array is padded

    # rescaling:
    array_standard_xy = np.zeros((target_shape[0], target_shape[1], shape_of_target_volume[2]), 'float32')
    for s in range(shape_of_target_volume[2]):
        array_standard_xy[:, :, s] = cv2.resize(array_intermediate[:, :, s], (target_shape[0], target_shape[1]), cv2.INTER_LANCZOS4)

    array_standard = np.zeros(target_shape, 'float32')
    for s in range(target_shape[0]):
        array_standard[s, :, :] = cv2.resize(array_standard_xy[s, :, :], (target_shape[1], target_shape[2]), cv2.INTER_LINEAR)

    return array_standard


def save_raw_arrays(patient_id, target_resolution=(334/512, 334/512, 1), target_shape=(512, 512, 512), compress=False):
    save_dict = Functions.get_father_dict() + '/arrays_raw/'
    print('processing:', patient_id)
    data_list, time_list_1 = read_in_CT.get_ct_array(patient_id, show=False)
    mask_list, time_list_2 = process_mha.get_mask_array(patient_id)
    time_list, shape_list, resolutions_list = read_in_CT.get_info(patient_id)
    assert time_list_1 == time_list_2 and time_list_2 == time_list
    num_scan = len(time_list)

    for t in range(num_scan):
        shape = list(target_shape)
        shape.append(2)
        raw_array = np.zeros(shape, 'float32')
        array_name = patient_id + '_' + str(time_list[t])
        if os.path.exists(save_dict + array_name + '.npy'):
            print('the file is exist:', array_name)
            continue
        raw_array[:, :, :, 0] = rescale_to_standard(data_list[t], resolutions_list[t], target_resolution, target_shape)
        raw_array[:, :, :, 1] = rescale_to_standard(mask_list[t], resolutions_list[t], target_resolution, target_shape)
        Functions.save_np_array(save_dict, array_name, raw_array, compress)
    return 0
