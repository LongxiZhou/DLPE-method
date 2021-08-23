import Tool_Functions.Functions as Functions
import numpy as np
import os
'''
the rescaled array should in shape: [x, y, z, data_channel + enhanced_channel + semantic_channel]
semantic_channel is the masks, which should in range [0, 1];
'''
semantic_channel = 1
data_channel = 1  # for rescaled array, CT has one data channel, while DCE-MRI have more. And the data_channel for
# samples usually greater than one if it used a slicing window.
enhanced_channel = 0
array_directory = '/home/zhoul0a/Desktop/air_tube_seg/arrays_raw/'
# array_directory is the directory where the rescaled array located.
output_directory = '/home/zhoul0a/Desktop/air_tube_seg/visualization/'
# where the visualization will be output.
signal_window = [-0.5, 0.5]  # the data_channel will be clipped by the signal_window


def get_mask_and_data(file_name):
    array_path = array_directory + file_name
    rescaled_array = np.load(array_path)
    print("the rescaled_array has shape:", np.shape(rescaled_array))
    assert np.shape(rescaled_array)[3] == semantic_channel + data_channel + enhanced_channel
    mid_channel = int(data_channel/2)
    data = rescaled_array[:, :, :, mid_channel]
    Functions.array_stat(data)
    data = np.clip(data, signal_window[0], signal_window[1])
    data = Functions.cast_to_0_1(data)
    print("the data is rescaled into [0, 1]")
    mask = rescaled_array[:, :, :, data_channel + enhanced_channel::]
    print("we have", np.shape(mask)[3], "number of semantics")
    # data has shape like [512, 512, 512]
    # mask has shape like [512, 512, 512, semantic]
    return data, mask


def slice_by_slice_plot(array_ct, save_dict='/home/zhoul0a/Desktop/prognosis_project/visualize/test/', direction='X', gray=True):
    assert len(np.shape(array_ct)) == 3
    array_shape = np.shape(array_ct)
    array = np.array(array_ct, 'float32')  # deepcopy the array_ct
    max_value = np.max(array)
    min_value = np.min(array)
    array[:, 0, 0] = max_value
    array[:, 0, array_shape[2]-1] = min_value  # the above two makes direction 'X' have same max and min
    array[0, :, 0] = max_value
    array[0, :, array_shape[2]-1] = min_value  # the above two makes direction 'Y' have same max and min
    array[0, 0, :] = max_value
    array[0, array_shape[1]-1, :] = min_value  # the above two makes direction 'Z' have same max and min
    if direction == 'X':
        for x in range(array_shape[0]):
            Functions.image_save(array[x, :, :], os.path.join(save_dict, str(x) + '.png'), gray=gray)
    if direction == 'Y':
        for y in range(array_shape[1]):
            Functions.image_save(array[:, y, :], os.path.join(save_dict, str(y) + '.png'), gray=gray)
    if direction == 'Z':
        for z in range(array_shape[2]):
            Functions.image_save(array[:, :, z], os.path.join(save_dict, str(z) + '.png'), gray=gray)


def visualize_mask_and_raw_array(mask, array_cut, save_dic, neglect_all_negative=True, clip=True):
    # mask has shape like [512, 512, 512]
    # array_cut has shape like [512, 512, 512] and should in range [0, 1]

    shape = np.shape(mask)

    merge = np.zeros([shape[0], shape[1] * 2, shape[2], 3], 'float32')

    merge[:, 0: shape[1], :, 0] = array_cut
    merge[:, 0: shape[1], :, 1] = array_cut
    merge[:, 0: shape[1], :, 2] = array_cut

    merge[:, shape[1]::, :, 0] = array_cut + mask
    merge[:, shape[1]::, :, 1] = array_cut - mask
    merge[:, shape[1]::, :, 2] = array_cut - mask

    if clip:
        merge = np.clip(merge, 0, 1)
        # assert np.min(merge) == 0 and np.max(merge) == 1

    if not os.path.exists(save_dic):
        os.makedirs(save_dic)

    for i in range(shape[2]):
        if neglect_all_negative:
            if np.sum(mask[:, :, i]) == 0:
                continue
        Functions.image_save(merge[:, :, i], save_dic + str(i), gray=False)


def highlight_mask(mask, array, channel='R', further_highlight=False, transparency=0.):
    """
    :param transparency: a float 0-1, controls the transparency of the highlight mask
    :param further_highlight: if True, means we add new semantic
    :param mask: 3D binary array
    :param array: 3D data array, in range [0, 1]; or 4D array like [512, 512, 512, 3] in range [0, 1]
    :param channel: R, G, B, ...
    :return: 4D array like [512, 512 * 2, 512, 3], in range [0, 1]
    """
    assert 0 <= transparency <= 1
    assert channel in ['R', 'G', 'B', 'Y', 'A', 'P']
    mask = np.array(mask, 'float32') * (1 - transparency)
    shape = np.shape(mask)
    if not further_highlight:
        assert shape == np.shape(array)
        merge = np.zeros([shape[0], shape[1] * 2, shape[2], 3], 'float32')
        merge[:, 0: shape[1], :, 0] = array
        merge[:, 0: shape[1], :, 1] = array
        merge[:, 0: shape[1], :, 2] = array

        if channel == 'R':  # red
            merge[:, shape[1]::, :, 0] = array + mask
            merge[:, shape[1]::, :, 1] = array - mask
            merge[:, shape[1]::, :, 2] = array - mask
        if channel == 'G':  # green
            merge[:, shape[1]::, :, 0] = array - mask
            merge[:, shape[1]::, :, 1] = array + mask
            merge[:, shape[1]::, :, 2] = array - mask
        if channel == 'B':  # blue
            merge[:, shape[1]::, :, 0] = array - mask
            merge[:, shape[1]::, :, 1] = array - mask
            merge[:, shape[1]::, :, 2] = array + mask
        if channel == 'Y':  # yellow
            merge[:, shape[1]::, :, 0] = array + mask
            merge[:, shape[1]::, :, 1] = array + mask
            merge[:, shape[1]::, :, 2] = array - mask
        if channel == 'A':  # azure
            merge[:, shape[1]::, :, 0] = array - mask
            merge[:, shape[1]::, :, 1] = array + mask
            merge[:, shape[1]::, :, 2] = array + mask
        if channel == 'P':  # purple
            merge[:, shape[1]::, :, 0] = array + mask
            merge[:, shape[1]::, :, 1] = array - mask
            merge[:, shape[1]::, :, 2] = array + mask
    else:
        merge = np.array(array)
        array = np.array(array[:, shape[1]::, :, :])
        if channel == 'R':  # red
            merge[:, shape[1]::, :, 0] = array[:, :, :, 0] + mask
            merge[:, shape[1]::, :, 1] = array[:, :, :, 1] - mask
            merge[:, shape[1]::, :, 2] = array[:, :, :, 2] - mask
        if channel == 'G':  # green
            merge[:, shape[1]::, :, 0] = array[:, :, :, 0] - mask
            merge[:, shape[1]::, :, 1] = array[:, :, :, 1] + mask
            merge[:, shape[1]::, :, 2] = array[:, :, :, 2] - mask
        if channel == 'B':  # blue
            merge[:, shape[1]::, :, 0] = array[:, :, :, 0] - mask
            merge[:, shape[1]::, :, 1] = array[:, :, :, 1] - mask
            merge[:, shape[1]::, :, 2] = array[:, :, :, 2] + mask
        if channel == 'Y':  # yellow
            merge[:, shape[1]::, :, 0] = array[:, :, :, 0] + mask
            merge[:, shape[1]::, :, 1] = array[:, :, :, 1] + mask
            merge[:, shape[1]::, :, 2] = array[:, :, :, 2] - mask
        if channel == 'A':  # azure
            merge[:, shape[1]::, :, 0] = array[:, :, :, 0] - mask
            merge[:, shape[1]::, :, 1] = array[:, :, :, 1] + mask
            merge[:, shape[1]::, :, 2] = array[:, :, :, 2] + mask
        if channel == 'P':  # purple
            merge[:, shape[1]::, :, 0] = array[:, :, :, 0] + mask
            merge[:, shape[1]::, :, 1] = array[:, :, :, 1] - mask
            merge[:, shape[1]::, :, 2] = array[:, :, :, 2] + mask

    return np.clip(merge, 0, 1)


def visualize_merged(merged_array, save_dict, neglect_all_negative=True, direction='Z'):
    if not os.path.exists(save_dict):
        os.makedirs(save_dict)
    shape = np.shape(merged_array)
    mask = np.array(merged_array[:, int(shape[1]/2)::, :, :] - merged_array[:, 0: int(shape[1]/2), :, :] > 0, 'float32')
    if direction == 'Z':
        for i in range(shape[2]):
            if neglect_all_negative:
                if np.sum(mask[:, :, i, :]) == 0:
                    continue
            Functions.image_save(merged_array[:, :, i], save_dict + str(i), gray=False)
    if direction == 'X':
        for i in range(shape[0]):
            if neglect_all_negative:
                if np.sum(mask[i, :, :, :]) == 0:
                    continue
            image = np.zeros([int(shape[1]/2), shape[2] * 2, 3], 'float32')
            image[:, 0: shape[2], :] = merged_array[i, 0: int(shape[1]/2), :, :]
            image[:, shape[2]::, :] = merged_array[i, int(shape[1]/2)::, :, :]
            Functions.image_save(image, save_dict + str(i), gray=False)
    if direction == 'Y':
        for i in range(int(shape[1]/2)):
            if neglect_all_negative:
                if np.sum(mask[:, i, :, :]) == 0:
                    continue
            image = np.zeros([int(shape[1]/2), shape[2] * 2, 3], 'float32')
            image[:, 0: shape[2], :] = merged_array[:, i, :, :]
            image[:, shape[2]::, :] = merged_array[:, i + int(shape[1]/2), :, :]
            Functions.image_save(image, save_dict + str(i), gray=False)


def visualize_one_array(file_name):
    data, mask = get_mask_and_data(file_name)
    num_semantic = np.shape(mask)[3]
    for i in range(num_semantic):
        visualize_mask_and_raw_array(mask[:, :, :, i], data, output_directory + file_name + '/semantic_' + str(i) + '/')


if __name__ == "__main__":

    array = np.load('/home/zhoul0a/Desktop/heart_seg/rescaled_ct_and_gt/xz0012_2020-05-01.npy')
    array[:, :, :, 0] = array[:, :, :, 0] + 0.5
    array_cut = np.clip(array, 0, 1)
    visualize_mask_and_raw_array(array[:, :, :, 1], array[:, :, :, 0], '/home/zhoul0a/Desktop/heart_seg/visualize/check_gt/')
    exit()
    slice_by_slice_plot(array_cut, '/home/zhoul0a/Desktop/prognosis_project/visualize/debug/', direction='Z')
    exit()

    scan_name = 'xgfh-2/xgfh-2_2020-07-29.npy'
    array_cut = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_data/'+scan_name[:-4]+'.npy') + 0.5
    array_cut = np.clip(array_cut, 0, 1)
    slice_by_slice_plot(array_cut, '/home/zhoul0a/Desktop/prognosis_project/visualize/test/'+scan_name[:-4]+'Z_clip/', direction='Z')
    exit()

    mask = np.load('/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_masks/infection_COVID-19/' + scan_name)['array']

    visualize_mask_and_raw_array(mask, array_cut, '/home/zhoul0a/Downloads/PA_PV/new_infection_COVID-19/' + scan_name[:-9] + '/')
