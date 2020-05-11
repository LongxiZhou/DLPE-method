import Functions
import numpy as np
import os


def visualize_mask_and_raw_array(mask, array_ct, save_dic='/home/zhoul0a/Desktop/COVID-19/Test/images/'):
    Functions.array_stat(array_ct)
    
    array_cut = np.clip(array_ct, -0.5, 0.5) + 0.5

    shape = np.shape(mask)

    merge = np.zeros([shape[0], shape[1] * 2, shape[2], 3], 'float32')

    merge[:, 0: shape[1], :, 0] = array_cut
    merge[:, 0: shape[1], :, 1] = array_cut
    merge[:, 0: shape[1], :, 2] = array_cut

    merge[:, shape[1]::, :, 0] = array_cut
    merge[:, shape[1]::, :, 1] = array_cut - mask
    merge[:, shape[1]::, :, 2] = array_cut - mask

    merge = np.clip(merge, 0, 1)

    assert np.min(merge) == 0 and np.max(merge) == 1

    if not os.path.exists(save_dic):
        os.makedirs(save_dic)

    for i in range(shape[2]):
        if np.sum(mask[:, :, i]) == 0:
            continue
        Functions.image_save(merge[:, :, i], save_dic + str(i), gray=False)
