import Tool_Functions.Functions as Functions
import os
import bintrees
import numpy as np
import pydicom


def stack_dcm_files_simplest(dic, show=True):

    dcm_file_names = os.listdir(dic)
    num_slices = len(dcm_file_names)
    if show:
        print("number_dcm_files:", num_slices)
    first_slice = Functions.load_dicom(os.path.join(dic, dcm_file_names[0]))[0]

    first_content = pydicom.read_file(os.path.join(dic, dcm_file_names[0]))
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    if show:
        print('the resolution for x, y, z in mm:', resolutions)
    rows, columns = first_slice.shape
    tree_instance = bintrees.AVLTree()
    slice_id_list = []
    array_3d = np.zeros([rows, columns, num_slices + 3], 'float32')
    for file in dcm_file_names:
        data_array, slice_id = Functions.load_dicom(os.path.join(dic, file))
        slice_id_list.append(slice_id)
        assert not tree_instance.__contains__(slice_id)
        tree_instance.insert(slice_id, slice_id)

        array_3d[:, :, num_slices - slice_id] = data_array
    assert np.max(slice_id_list) - np.min(slice_id_list) + 1 == len(slice_id_list)
    array_3d = array_3d[:, :, num_slices - np.max(slice_id_list): num_slices - np.min(slice_id_list) + 1]
    if show:
        print('the array corresponds to a volume of:',
              rows*resolutions[0], columns*resolutions[1], num_slices*resolutions[2])
        Functions.array_stat(array_3d)
        print('stack complete!')
    return array_3d, resolutions


def stack_dcm_files(dic, show=True, wc_ww=(-600, 1600), use_default=True):
    # the dictionary like '/home/zhoul0a/CT_slices_for_patient_alice/'
    # wc_ww should be a tuple like (-600, 1600) if you want to to assign wc_ww.
    # return a 3D np array with shape [Rows, Columns, Num_Slices], and the resolution of each axis
    dcm_file_names = os.listdir(dic)
    num_slices = len(dcm_file_names)
    first_slice = Functions.load_dicom(os.path.join(dic, dcm_file_names[0]))[0]
    try:
        wc, ww = Functions.wc_ww(os.path.join(dic, dcm_file_names[0]))
    except:
        print("no ww and wc, use default")
        wc, ww = wc_ww
    if (wc < -800 or wc > -400 or ww > 1800 or ww < 1400) and use_default:
        print("the original wc, ww is:", wc, ww, "which is strange, we use default.")
    if wc_ww is not None:
        wc, ww = wc_ww
    print('the window center and window width are:')
    print("\n### ", wc, ",", ww, '###\n')
    first_content = pydicom.read_file(os.path.join(dic, dcm_file_names[0]))
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    if show:
        print('the resolution for x, y, z in mm:', resolutions)
    rows, columns = first_slice.shape
    tree_instance = bintrees.AVLTree()
    array_3d = np.zeros([rows, columns, num_slices], 'int32')
    slice_id_list = []
    for file in dcm_file_names:
        data_array, slice_id = Functions.load_dicom(os.path.join(dic, file))
        slice_id_list.append(slice_id)
        slice_id -= 1
        assert not tree_instance.__contains__(slice_id)
        tree_instance.insert(slice_id, slice_id)
        array_3d[:, :, num_slices - slice_id - 1] = data_array
    if show:
        print('the array corresponds to a volume of:', rows*resolutions[0], columns*resolutions[1], num_slices*resolutions[2])
    Functions.array_stat(array_3d)
    array_3d -= wc
    array_3d = array_3d / ww  # cast the lung signal into -0.5 to 0.5
    print('stack complete!')
    slice_id_list.sort()
    print(slice_id_list)

    return array_3d, resolutions


def get_ct_array(patient_id, show=False):
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/check_format/patients/' + patient_id + '/'
    time_points = os.listdir(top_dic)
    array_list = []
    for time in time_points:
        array, _ = stack_dcm_files(top_dic + time + '/Data/raw_data/', show)
        array_list.append(array)
    return array_list, time_points


def get_info(patient_id, show=False):
    if show:
        print('get information for patient:', patient_id)
    top_dic = os.path.abspath(os.path.join(os.getcwd(), '..')) + '/check_format/patients/' + patient_id + '/'
    time_points = os.listdir(top_dic)
    if show:
        print('we have these time points:', time_points)
    resolutions_list = []  # the elements are [x1, y1, z1], [x2, y2, z2], ...
    shape_list = []
    for time in time_points:
        data_dict = top_dic + time + '/Data/raw_data/'
        dcm_list = os.listdir(data_dict)
        num_slices = len(dcm_list)
        first_slice = Functions.load_dicom(data_dict + dcm_list[0])[0]
        rows, columns = first_slice.shape
        first_content = pydicom.read_file(data_dict + dcm_list[0])
        resolutions = first_content.PixelSpacing
        resolutions.append(first_content.SliceThickness)
        shape = [rows, columns, num_slices]
        shape_list.append(shape)
        resolutions_list.append(resolutions)
        wc, ww = Functions.wc_ww(data_dict + dcm_list[0])
        if show:
            print('time point', time, 'has shape', shape, ', resolution', resolutions, ', wc ww', wc, ww)
    if show:
        print('\n')
    return time_points, shape_list, resolutions_list


if __name__ == '__main__':
    image, _ = Functions.load_dicom('/media/zhoul0a/641F-617D/Radiology/1276068817/CT/20181223/rawdata/1487EAC7', show=True)
    image = np.clip(image, 0, 366)
    Functions.image_show(image, gray=True)

    array, resolution = stack_dcm_files('/media/zhoul0a/641F-617D/Radiology/1277026936/CT/20181206/rawdata/', show=True)
    "1277026936_20181206.npz"

    print(resolution)
    print(np.shape(array))
    array = np.clip(array, -0.5, 0.5)
    Functions.image_show(array[:, :, 110], gray=True)
