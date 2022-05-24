import os
import sys
import random
import numpy as np
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import models.Unet_2D.test as test_model
import Tool_Functions.Functions as Functions

array_info = {
    "resolution": (1, 1, 1),
    "data_channel": 1,
    "enhanced_channel": 0,
    "window": (-1, 0, 1),
    "positive_semantic_channel": (0, ),
    "output_channels": 2,
    "mute_output": True,
    "wrong_scan": ['xwqg-A00085', 'xwqg-A00121', 'xwqg-B00027', 'xwqg-B00034'],
    "init_features": 16
}


def check_point_path_generator(check_point_dict, file_name, direction):
    """
    we need this function as we may use different training strategy like 5-fold, 10-fold.
    :param check_point_dict: directory where the check_points, like: '/ibex/check_points/'
    :param file_name: the file name of the scan, like xwqg-A00121_2019-05-15.npy
    :param direction: like 'X', 'Y', 'Z'
    :return: freq path of the correct check_point_file
    """
    test_id = int(file_name.split('_')[0][-1]) % 5
    return check_point_dict + direction + '/' + str(test_id) + '_saved_model.pth'


def remove_wrong_scans(file_name_list):
    wrong_scan = array_info["wrong_scan"]
    if wrong_scan is not None:
        print("the wrong scans are:", wrong_scan)
        for file_name in file_name_list:
            patient_id = file_name.split('_')[0]
            if patient_id in wrong_scan:
                file_name_list.remove(file_name)
                print("removed wrong scan:", file_name)
    return file_name_list


def three_way_prediction(scan_dict, scan_name, check_point_dict, threshold=2., evaluate_sub_model=False, batch_size=64,
                         evaluate=False, gt_channel=3):
    """
    :param evaluate: whether evaluate the dice, recall and precision
    :param batch_size: the batch_size when predicting.
    :param evaluate_sub_model: whether evaluate the performance of single direction.
    :param threshold: when combine, the threshold of considered as positive. When threshold is None, return sum of three
    directions.
    :param scan_dict: directory where the scan stored, like: '/ibex/data/scan_dict/'
    :param scan_name: the file name of the scan, like xwqg-A00121_2019-05-15.npy
    :param check_point_dict: directory check points saved, like '/ibex/scratch/projects/air_tube_seg/check_points/'
    :param gt_channel: the ground truth channel of the rescaled_data
    :return: prediction masks; or prediction masks, recall, precision, dice
    """
    rescaled_array = np.load(scan_dict + scan_name)
    check_point_path_X = check_point_path_generator(check_point_dict, scan_name, 'X')
    check_point_path_Y = check_point_path_generator(check_point_dict, scan_name, 'Y')
    check_point_path_Z = check_point_path_generator(check_point_dict, scan_name, 'Z')

    prediction_X = test_model.predict_one_scan_binary_class(rescaled_array, 'X', check_point_path_X, array_info,
                                                            evaluate=evaluate_sub_model, batch_size=batch_size)
    prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array, 'Y', check_point_path_Y, array_info,
                                                            evaluate=evaluate_sub_model, batch_size=batch_size)
    prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array, 'Z', check_point_path_Z, array_info,
                                                            evaluate=evaluate_sub_model, batch_size=batch_size)
    if threshold is not None:
        prediction = np.array(prediction_X + prediction_Y + prediction_Z > threshold, 'float32')
    else:
        return prediction_X + prediction_Y + prediction_Z

    if evaluate:
        recall, precision, dice = Functions.evaluate(prediction, rescaled_array[:, :, :, gt_channel], threshold=0)
        # if not array_info["mute_output"]:
        print("overall recall, precision, dice for", scan_name, "is", recall, precision, dice)
        return prediction, recall, precision, dice

    return prediction


def two_stage_three_way_enhance_two(scan_dict, scan_name, check_point_dict_list,
                                    threshold=2., batch_size=64, evaluate=False, gt_channel=1):
    """
    the first enhance is with threshold 2.0 (recall 0.95, precision 0.56), the second enhance is with threshold 2.99
    (recall 0.65, precision 0.95)
    :param scan_dict: directory where the scan stored, like: '/ibex/data/scan_dict/'
    :param scan_name: the file name of the scan, like xwqg-A00121_2019-05-15.npy
    :param check_point_dict_list: [first_round check point dict, second_round check point dict]
    :param threshold: threshold for the second stage
    :param batch_size: batch_size when predicting
    :param evaluate: if false, return prediction array, else, return prediction, recall, precision and dice
    :param gt_channel: the ground truth channel of the rescaled array
    :return: prediction array or, prediction, recall, precision, dice
    """
    check_point_dict_stage_one = check_point_dict_list[0]
    check_point_dict_stage_two = check_point_dict_list[1]
    rescaled_array = np.load(scan_dict + scan_name)
    shape_original = np.shape(rescaled_array)
    prediction_stage_one = three_way_prediction(scan_dict, scan_name, check_point_dict_stage_one, threshold=None)
    enhanced_channel_one = np.array(prediction_stage_one > 2., 'float32')
    enhanced_channel_two = np.array(prediction_stage_one > 2.99, 'float32')
    rescaled_array_enhance_two = np.zeros((shape_original[0], shape_original[1], shape_original[2],
                                           shape_original[3] + 2), 'float32')

    rescaled_array_enhance_two[:, :, :, 0] = rescaled_array[:, :, :, 0]  # data
    rescaled_array_enhance_two[:, :, :, 1] = enhanced_channel_one  # high recall
    rescaled_array_enhance_two[:, :, :, 2] = enhanced_channel_two  # high precision
    if evaluate and shape_original[3] > 1:
        rescaled_array_enhance_two[:, :, :, 3] = rescaled_array[:, :, :, gt_channel]  # ground truth

    check_point_path_X = check_point_path_generator(check_point_dict_stage_two, scan_name, 'X')
    check_point_path_Y = check_point_path_generator(check_point_dict_stage_two, scan_name, 'Y')
    check_point_path_Z = check_point_path_generator(check_point_dict_stage_two, scan_name, 'Z')

    old_enhance_info = array_info["enhanced_channel"]
    array_info["enhanced_channel"] = 2

    prediction_X = test_model.predict_one_scan_binary_class(rescaled_array_enhance_two, 'X', check_point_path_X, array_info,
                                                            evaluate=evaluate, batch_size=batch_size)
    prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array_enhance_two, 'Y', check_point_path_Y, array_info,
                                                            evaluate=evaluate, batch_size=batch_size)
    prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array_enhance_two, 'Z', check_point_path_Z, array_info,
                                                            evaluate=evaluate, batch_size=batch_size)
    array_info["enhanced_channel"] = old_enhance_info
    if threshold is None:
        return prediction_X + prediction_Y + prediction_Z
    else:
        prediction = np.array(prediction_X + prediction_Y + prediction_Z > threshold, 'float32')

    if evaluate and shape_original[3] > 1:
        recall, precision, dice = Functions.evaluate(prediction, rescaled_array[:, :, :, gt_channel], threshold=0)
        # if not array_info["mute_output"]:
        print("overall recall, precision, dice for", scan_name, "is", recall, precision, dice)
        return prediction, recall, precision, dice

    return prediction


def get_optimal_threshold(rescale_array_dict, check_point_dict_list, test_ratio, prediction_model=three_way_prediction,
                          search_paras=(1.5, 2.4, 0.1), gt_channel=3):
    if len(check_point_dict_list) == 1:
        check_point_dict_list = check_point_dict_list[0]
    file_name_list = os.listdir(rescale_array_dict)
    file_name_list = remove_wrong_scans(file_name_list)
    random.shuffle(file_name_list)
    num_predict = int(test_ratio * len(file_name_list))
    search_start = search_paras[0]
    search_end = search_paras[1]
    search_step = search_paras[2]
    threshold = search_start
    recall_list_threshold = []
    precision_list_threshold = []
    dice_list_threshold = []
    threshold_list = []
    while threshold < search_end:
        threshold_list.append(threshold)
        recall_list = []
        precision_list = []
        dice_list = []
        print("predicting with threshold:", threshold)
        predicted = 0
        for file_name in file_name_list[0: num_predict]:
            print("number scan left:", num_predict - predicted)
            print("processing:", file_name, "threshold:", threshold)
            prediction, recall, precision, dice = prediction_model(rescale_array_dict, file_name, check_point_dict_list,
                                                                   threshold=threshold, evaluate=True, gt_channel=gt_channel)
            recall_list.append(recall)
            precision_list.append(precision)
            dice_list.append(dice)
            print("current mean, recall, precision, dice", np.mean(recall_list), np.mean(precision_list), np.mean(dice_list))
            predicted += 1
        print('\n###################\n')
        print("the performance under threshold", threshold, 'is')
        print("recall, precision, dice:", np.mean(recall_list), np.mean(precision_list), np.mean(dice_list))
        recall_list_threshold.append(np.mean(recall_list))
        precision_list_threshold.append(np.mean(precision_list))
        dice_list_threshold.append(np.mean(dice_list))
        print('\n###################\n')
        threshold += search_step
    print('\n###################\n')
    print("thresholds:")
    print(threshold_list)
    print("recall, precision, dice on the data set:")
    print(recall_list_threshold)
    print(precision_list_threshold)
    print(dice_list_threshold)


def three_way_get_enhanced_channel_and_save(rescaled_array_dict, check_point_dict, save_dict, batch_size=48):
    # the inner enhance model can reach recall of 0.98 with dice 0.5, which can be the enhanced channels.
    """
    :param rescaled_array_dict: stores the original rescaled arrays, with shape like [512, 512, 512, data_channel + s]
    :param check_point_dict: the inner enhance models, which produce the enhanced channels.
    :param save_dict: where the new rescaled_array with enhanced channels stored
    :param batch_size: the batch_size when calculate enhanced channels
    :return: None
    """
    file_name_list = os.listdir(rescaled_array_dict)
    print("there are:", len(file_name_list), "in this directory")
    file_name_list = remove_wrong_scans(file_name_list)
    num_files = len(file_name_list)
    for i in range(num_files):
        file_name = file_name_list[i]
        print("processing:", file_name, num_files - i, 'files left')
        if os.path.exists('/ibex/scratch/projects/c2052/air_tube_seg/enhanced_rescaled_array/' + file_name):
            if os.path.exists(save_dict + file_name):
                print("already processed")
                continue
            print("cutting...")
            array = np.load('/ibex/scratch/projects/c2052/air_tube_seg/enhanced_rescaled_array/' + file_name)
            new_array = np.zeros([512, 512, 512, 4], 'float32')
            new_array[:, :, :, 0] = array[:, :, :, 0]  # data channel
            pre_x = array[:, :, :, 1]
            pre_y = array[:, :, :, 2]
            pre_z = array[:, :, :, 3]
            new_array[:, :, :, 1] = np.array(pre_x + pre_y + pre_z > 2.0, 'float32')  # high recall channel
            new_array[:, :, :, 2] = np.array(pre_x + pre_y + pre_z > 2.99, 'float32')  # high precision channel
            new_array[:, :, :, 3] = array[:, :, :, 4]  # gt channel
            Functions.save_np_array(save_dict, file_name, new_array, False)
            continue
        rescaled_array = np.load(rescaled_array_dict + file_name)
        old_shape = np.shape(rescaled_array)
        print("old shape:", old_shape)

        check_point_path_X = check_point_path_generator(check_point_dict, file_name, 'X')
        check_point_path_Y = check_point_path_generator(check_point_dict, file_name, 'Y')
        check_point_path_Z = check_point_path_generator(check_point_dict, file_name, 'Z')

        prediction_X = test_model.predict_one_scan_binary_class(rescaled_array, 'X', check_point_path_X, array_info,
                                                                evaluate=False, batch_size=batch_size)
        prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array, 'Y', check_point_path_Y, array_info,
                                                                evaluate=False, batch_size=batch_size)
        prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array, 'Z', check_point_path_Z, array_info,
                                                                evaluate=False, batch_size=batch_size)
        combined = np.array(prediction_X + prediction_Y + prediction_Z > 1.5, 'float32')
        enhanced_rescaled_array = np.zeros((old_shape[0], old_shape[1], old_shape[2], old_shape[3] + 1), 'float32')
        enhanced_rescaled_array[:, :, :, 0] = rescaled_array[:, :, :, 0]  # data channel
        enhanced_rescaled_array[:, :, :, 1] = combined
        # enhanced_rescaled_array[:, :, :, 2] = prediction_Y
        # enhanced_rescaled_array[:, :, :, 3] = prediction_Z
        enhanced_rescaled_array[:, :, :, 2] = rescaled_array[:, :, :, 1]  # semantic channel

        print("saving enhanced_rescaled_array...\n")
        Functions.save_np_array(save_dict, file_name, enhanced_rescaled_array, False)


get_optimal_threshold('/ibex/scratch/projects/c2052/air_tube_seg/arrays_raw/',
                      ['/ibex/scratch/projects/c2052/air_tube_seg/check_points/balance_weight_array/',
                       '/ibex/scratch/projects/c2052/air_tube_seg/check_points_enhanced_two_light/balance_weight_array_rim_enhance/'], 0.05,
                      two_stage_three_way_enhance_two, search_paras=(1.6, 3.0, 0.2))

exit()
three_way_get_enhanced_channel_and_save('/ibex/scratch/projects/c2052/air_tube_seg/arrays_raw/',
                                        '/ibex/scratch/projects/c2052/air_tube_seg/check_points/balance_weight_array/',
                                        '/ibex/scratch/projects/c2052/air_tube_seg/enhanced_two_channel_rescaled_array/')
exit()
