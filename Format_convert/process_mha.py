import SimpleITK as sitk
import numpy as np
import os


def read_in_mha(path):

    ar = sitk.ReadImage(path)

    mask = sitk.GetArrayFromImage(ar)

    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    mask = np.array(mask > 0, 'int32')
    # print(np.shape(mask))

    return mask  # (x, y, z)


def get_mask_array(patient_id, lung_mask=False):
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/check_format/patients/' + patient_id + '/'
    time_points = os.listdir(top_dic)
    array_list = []
    for time in time_points:
        if not lung_mask:
            print('loading mha')
            array = read_in_mha(top_dic + time + '/Data/ground_truth/LI.mha')
        else:
            print('loading mha')
            array = read_in_mha(top_dic + time + '/Data/ground_truth/右肺(分割).mha')
            array = array + read_in_mha(top_dic + time + '/Data/ground_truth/左肺(分割).mha')
        array_list.append(array)
    return array_list, time_points
