import Tool_Functions.Functions as Functions
import format_convert.spatial_normalize as normalize
import format_convert.dcm_np_converter as converter
import Tool_Functions.id_time_generator as generator
import os
import numpy as np

"""
dataset should with directory root_dict/patient-id/time/Data/raw_data(ground_truth)/
ground truth must in .mha
save rescaled CT and rescaled GT separately

##############################################
# change this semantic list for new data set
##############################################
"""
semantic_list = ['fjj.mha']
# semantic_list = ['xb.mha', 'fdm.mha', 'fjm.mha', 'artery.mha', 'vein.mha']


def rescale_ct_gt_all_parallel(root_dict, save_dict_ct, save_dict_gt, parallel_count=24):
    """
    :param root_dict: stores the standardized data, root_dict/patient-id/time/Data/raw_data(ground_truth)/
    :param save_dict_ct: where to save rescaled ct: save_dict_ct/patient-id_time.npy
    :param save_dict_gt: where to save rescaled gt: save_dict_gt/patient-id_time.npz, [512, 512, 512, num_semantics],
    the order is the same with os.listdir(root_dict/patient-id/time/Data/ground_truth)
    :param parallel_count: how many programs run simultaneously
    :return: None
    """
    input_list = []
    for i in range(parallel_count):
        input_list.append([root_dict, save_dict_ct, save_dict_gt, parallel_count, i])

    Functions.func_parallel(ct_gt_rescale_one_thread, input_list, 8)


def ct_gt_rescale_one_thread(inputs):
    root_dict, save_dict_ct, save_dict_gt, parallel_count, para_id = inputs
    print('\n\n\n')

    id_time_list = generator.return_all_tuples_for_original_data(root_dict)
    print("there are:", len(id_time_list), "scans")
    print(id_time_list)
    for patient_id, time in id_time_list[para_id::parallel_count]:
        print("processing", patient_id, time)
        if os.path.exists(os.path.join(save_dict_ct, patient_id + '_' + time + '.npy')):
            print("processed")
            continue
        dcm_dict = root_dict + patient_id + '/' + time + '/Data/raw_data/'

        ct_array = converter.dcm_to_signal_rescaled(dcm_dict, wc_ww=(-600, 1600))
        resolution = converter.get_original_resolution(dcm_dict)
        rescaled_ct = normalize.rescale_to_standard(ct_array, resolution)

        gt_dict = root_dict + patient_id + '/' + time + '/Data/ground_truth/'

        num_gt = len(semantic_list)
        gt_mask = np.zeros([512, 512, 512, num_gt])
        for semantic in range(num_gt):
            name = semantic_list[semantic]
            print("processing gt", name)
            path = gt_dict + name
            original = Functions.read_in_mha(path)
            rescaled_gt = normalize.rescale_to_standard(original, resolution)
            gt_mask[:, :, :, semantic] = rescaled_gt
        if num_gt == 1:
            Functions.save_np_array(save_dict_gt, patient_id + '_' + time, gt_mask[:, :, :, 0], compress=True)
        else:
            Functions.save_np_array(save_dict_gt, patient_id + '_' + time, gt_mask, compress=True)
        Functions.save_np_array(save_dict_ct, patient_id + '_' + time, rescaled_ct, compress=False)


def rescaled_arrays_parallel(root_dict, save_dict, parallel_count=24):
    """
    :param root_dict: stores the standardized data, root_dict/patient-id/time/Data/raw_data(ground_truth)/
    :param save_dict: where to save rescaled gt: save_dict_gt/patient-id_time.npz, [512, 512, 512, num_semantics + 1],
    the order is the same with os.lisdir(root_dict/patient-id/time/Data/ground_truth)
    :param parallel_count: how many programs run simultaneously
    :return: None
    """
    input_list = []
    for i in range(parallel_count):
        input_list.append([root_dict, save_dict, parallel_count, i])

    Functions.func_parallel(rescaled_arrays_one_thread, input_list, 8)


def rescaled_arrays_one_thread(inputs):
    root_dict, save_dict, parallel_count, para_id = inputs
    print('\n\n\n')
    id_time_list = generator.return_all_tuples_for_original_data(root_dict)
    print("there are:", len(id_time_list), "scans")
    print(id_time_list)
    for patient_id, time in id_time_list[para_id::parallel_count]:
        print("processing", patient_id, time)
        if os.path.exists(os.path.join(save_dict, patient_id + '_' + time + '.npz')):
            print("processed")
            continue
        dcm_dict = root_dict + patient_id + '/' + time + '/Data/raw_data/'

        ct_array = converter.dcm_to_signal_rescaled(dcm_dict)
        resolution = converter.get_original_resolution(dcm_dict)
        rescaled_ct = normalize.rescale_to_standard(ct_array, resolution)

        gt_dict = root_dict + patient_id + '/' + time + '/Data/ground_truth/'
        gt_file_name_list = os.listdir(gt_dict)
        num_gt = len(gt_file_name_list)
        #  assert num_gt == 2  # this is for airways vessel seg
        rescaled_array = np.zeros([512, 512, 512, 2])  # do not distinguish artery and vein
        rescaled_array[:, :, :, 0] = rescaled_ct
        for semantic in range(num_gt):
            name = gt_file_name_list[semantic]
            print("processing gt", name)
            path = gt_dict + name
            original = Functions.read_in_mha(path)
            rescaled_gt = normalize.rescale_to_standard(original, resolution)
            rescaled_array[:, :, :, 1] = rescaled_array[:, :, :, 1] + rescaled_gt
        rescaled_array[:, :, :, 1] = np.clip(rescaled_array[:, :, :, 1], 0, 1)

        Functions.save_np_array(save_dict, patient_id + '_' + time, rescaled_array, compress=True)


if __name__ == "__main__":
    rescale_ct_gt_all_parallel('/home/zhoul0a/Desktop/pulmonary nodules/ct_and_gt/',
                               '/home/zhoul0a/Desktop/pulmonary nodules/rescaled_array/',
                               '/home/zhoul0a/Desktop/pulmonary nodules/rescaled_gt/', parallel_count=1)
