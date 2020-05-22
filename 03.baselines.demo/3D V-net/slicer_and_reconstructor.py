import numpy as np



def slice_3d_into_sub_arrays(array_3d, size_of_sample=(128, 128, 128), stride=(32, 32, 32), neglect_all_negative=True):
    '''
    :param array_3d: should be in shape [512, 512, 512, 2] or [512, 512, 512]
    :param size_of_sample: the size of samples for 3D networks.
    :param stride: the stride when slicing array_3d
    :param neglect_all_negative: True for generating training samples, which will only return cubes containing infection
    :return: a list of np_arrays, float32, size: (size_of_sample, 2) or size_of_sample;
             a float, equals to: number_normal/number_infection
    '''

    list_of_arrays = []
    list_of_array_corners = []
    array_shape = np.shape(array_3d)
    x = 0
    while x + size_of_sample[0] <= array_shape[0]:
        y = 0
        while y + size_of_sample[1] <= array_shape[1]:
            z = 0
            while z + size_of_sample[2] <= array_shape[2]:
                list_of_array_corners.append((x, y, z))
                z += stride[2]
            y += stride[1]
        x += stride[0]

    for array_corner in list_of_array_corners:
        sub_cube = slicer(array_3d, array_corner, size_of_sample)
        if neglect_all_negative:
            if np.sum(sub_cube[:, :, :, 1]) < 10:
                continue
        list_of_arrays.append(sub_cube)

    infections = 0

    if len(array_shape) > 3:
        class_weights=[]
        for sub_cube in list_of_arrays:
            infections = np.sum(sub_cube[:, :, :, 1])
            assert infections>=0
            if infections<1e-6:
                class_weight = 1
            else:
                class_weight=min((size_of_sample[0]*size_of_sample[1]*size_of_sample[2]-infections)/(infections),100)
            class_weights.append(class_weight)
    else:
        class_weights = None

    return list_of_arrays, class_weights


def reconstructor(list_of_arrays, original_shape=(512, 512, 512)):
    '''
    :param list_of_arrays: the return list of slice_3d_into_sub_arrays(array_3d, size_of_sample, size_of_sample, False)
    :param original_shape: a tuple or a list, equals to np.shape(array_3d)[0:3]
    :return: an array, equals to array_3d
    '''
    size_of_sample = np.shape(list_of_arrays[0])
    stride = size_of_sample
    list_of_array_corners = []
    array_shape = original_shape
    x = 0
    while x + size_of_sample[0] <= array_shape[0]:
        y = 0
        while y + size_of_sample[1] <= array_shape[1]:
            z = 0
            while z + size_of_sample[2] <= array_shape[2]:
                list_of_array_corners.append((x, y, z))
                z += stride[2]
            y += stride[1]
        x += stride[0]

    reconstructed_array = np.zeros(original_shape, 'float32')

    num_of_arrays = len(list_of_arrays)
    assert num_of_arrays == len(list_of_array_corners)
    print('we have:', num_of_arrays, 'sub arrays')

    for i in range(num_of_arrays):
        array_corner = list_of_array_corners[i]
        sub_cube = list_of_arrays[i]
        reconstructed_array[
            array_corner[0]:array_corner[0]+size_of_sample[0],
            array_corner[1]:array_corner[1]+size_of_sample[1],
            array_corner[2]:array_corner[2]+size_of_sample[2],
        ] = sub_cube

    return reconstructed_array


def slicer(array_3d, nearest_corner, size_of_sample):
    '''
    :param array_3d: a np array with dim >= 3
    :param nearest_corner: a tuple or list, length=3
    :param size_of_sample: a tuple or list, length=3
    :return: a np_array with shape equals to size_of_sample
    '''
    return array_3d[
           nearest_corner[0]:nearest_corner[0]+size_of_sample[0],
           nearest_corner[1]:nearest_corner[1]+size_of_sample[1],
           nearest_corner[2]:nearest_corner[2]+size_of_sample[2]]


'''
# Example of test this code:

import numpy as np

import visualize_mask_and_raw

import slicer_and_reconstructor

import Functions


data = np.load('/home/zhoul0a/Desktop/COVID-19/arrays_raw/xgfy-A000010_2020-03-03.npy')

mask = data[:,:,:,1]
ct_scan = data[:,:,:,0]

array_list, weight = slicer_and_reconstructor.slice_3d_into_sub_arrays(ct_scan, (128, 128, 128), (128, 128, 128), False)

reconstructed_array = slicer_and_reconstructor.reconstructor(array_list, (512, 512, 512))

visualize_mask_and_raw.visualize_mask_and_raw_array(mask, reconstructed_array, save_dic='/home/zhoul0a/Desktop/COVID-19/Test/temp/')
'''