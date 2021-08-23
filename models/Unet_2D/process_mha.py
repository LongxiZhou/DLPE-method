import SimpleITK as sitk
import numpy as np
import read_in_CT
import os
import Functions

#  check the whether the ground truth has same shape with mask
current_dict = os.getcwd()
report_path = current_dict + '/problems_report'


def read_in_mha(path):
    ar = sitk.ReadImage(path)
    mask = sitk.GetArrayFromImage(ar)
    mask = np.swapaxes(mask, 0, 2)
    mask = np.swapaxes(mask, 0, 1)
    if not (np.max(mask) == 1 and np.min(mask) == 0):
        f = open(report_path, 'a')
        f.write('\n\nthe mask of this path is not a mask: ' + path)
        f.close()
    mask = np.array(mask > 0, 'int32')
    return mask  # (x, y, z)


def check_mask_and_raw(data_dict):
    mha_path = data_dict + 'ground_truth/LI.mha'
    try:
        mask = read_in_mha(mha_path)
    except:
        f = open(report_path, 'a')
        f.write('\n\nthe mask is not exist from this path: ' + mha_path)
        f.close()
        return 0
    try:
        array_ct, _ = read_in_CT.stack_dcm_files(data_dict + 'raw_data/')
    except:
        f = open(report_path, 'a')
        f.write('\n\ncould not load dcm files from this path: ' + data_dict + 'raw_data/')
        f.close()
        return 0
    if not np.shape(mask) == np.shape(array_ct):
        f = open(report_path, 'a')
        f.write('\n\nthe mask and raw data is not conform,\n')
        f.write('mask has shape:' + str(np.shape(mask)) + ' and data has shape:' + str(np.shape(array_ct)))
        f.write('\nplease check the .mha and .dcm files in: ' + data_dict)
        f.close()
        return 0
    return 1
