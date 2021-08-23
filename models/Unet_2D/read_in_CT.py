import os
import bintrees
import numpy as np
import pydicom
import SimpleITK as sitk


def load_dicom(path, show=False):
    # return a numpy array of the dicom file, and the slice number
    if show:
        content = pydicom.read_file(path)
        print(content)

    ds = sitk.ReadImage(path)

    img_array = sitk.GetArrayFromImage(ds)

    #  frame_num, width, height = img_array.shape

    return img_array[0, :, :], pydicom.read_file(path)['InstanceNumber'].value


def stack_dcm_files(dic):
    # the dictionary like '/home/zhoul0a/CT_slices_for_patient_alice/'
    # return a 3D np array with shape [Rows, Columns, Num_Slices], and the resolution of each axis: (0.625, 0.625, 0.9)
    dcm_file_names = os.listdir(dic)
    num_slices = len(dcm_file_names)
    first_slice = load_dicom(dic+dcm_file_names[0])[0]
    first_content = pydicom.read_file(dic+dcm_file_names[0])
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    print('the resolution for x, y, z in mm:', resolutions)
    rows, columns = first_slice.shape
    tree_instance = bintrees.AVLTree()
    array_3d = np.zeros([rows, columns, num_slices], 'int32')
    for file in dcm_file_names:
        data_array, slice_id = load_dicom(dic+file)
        assert not tree_instance.__contains__(slice_id)
        tree_instance.insert(slice_id, slice_id)
        array_3d[:, :, num_slices - slice_id] = data_array
    print('the array corresponds to a volume of:', rows*resolutions[0], columns*resolutions[1], num_slices*resolutions[2])
    return array_3d, resolutions


def stack_dcm_files_by_file_name(dic):
    # the dictionary like '/home/zhoul0a/CT_slices_for_patient_alice/'
    # return a 3D np array with shape [Rows, Columns, Num_Slices], and the resolution of each axis: (0.625, 0.625, 0.9)
    dcm_file_names = os.listdir(dic)
    num_slices = len(dcm_file_names)
    first_slice = load_dicom(dic+dcm_file_names[0])[0]
    first_content = pydicom.read_file(dic+dcm_file_names[0])
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    print('the resolution for x, y, z in mm:', resolutions)
    rows, columns = first_slice.shape
    tree_instance = bintrees.AVLTree()
    array_3d = np.zeros([rows, columns, num_slices], 'int32')
    for file in dcm_file_names:
        data_array, slice_id = load_dicom(dic+file)
        slice_id = int(file[-5]) + 10 * int(file[-6]) + 100 * int(file[-7]) - 1
        print(file, slice_id)
        if tree_instance.__contains__(slice_id):
            continue
        tree_instance.insert(slice_id, slice_id)
        array_3d[:, :, num_slices - slice_id] = data_array
    print('the array corresponds to a volume of:', rows*resolutions[0], columns*resolutions[1], num_slices*resolutions[2])
    return array_3d, resolutions


def get_information_for_dcm(path):
    array_dcm = load_dicom(path)[0]
    rows, columns = array_dcm.shape
    first_content = pydicom.read_file(path)
    resolutions = first_content.PixelSpacing
    resolutions.append(first_content.SliceThickness)
    return rows, columns, resolutions
