import os
import sys
import random
import numpy as np

sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import models.Unet_2D.test as test_model
import Tool_Functions.Functions as Functions

ibex = False
if not ibex:
    print("not ibex")
    import os
    top_directory = '/home/zhoul0a/Desktop/'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # use two V100 GPU
else:
    top_directory = '/ibex/scratch/projects/c2052/'

array_info = {
    "resolution": (1, 1, 1),
    "data_channel": 1,
    "enhanced_channel": 2,
    "window": (-1, 0, 1),
    "positive_semantic_channel": (0,),
    "output_channels": 2,  # the output channel of the model
    "mute_output": True,
    "wrong_scan": None,  # ['xwqg-A00085', 'xwqg-A00121', 'xwqg-B00027', 'xwqg-B00034'],
    "init_features": 16
}


def check_point_path_generator(check_point_dict, file_name, direction):
    """
    we need this function as we may use different training strategy like 5-fold, 10-fold.
    :param check_point_dict: directory where the check_points, like: '/ibex/check_points/direction/i_saved_model.pth'
    :param file_name: the file name of the scan, like xwqg-A00121_2019-05-15.npy
    :param direction: like 'X', 'Y', 'Z'
    :return: a path of the correct check_point_file
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
    :param scan_dict: directory where the scan stored, like: '/ibex/data/scan_dict/patient-id_time.npy'
    :param scan_name: the file name of the scan, patient-id_time.npy like xwqg-A00121_2019-05-15.npy
    :param check_point_dict: directory check points saved, like '/ibex/scratch/projects/air_tube_seg/check_points/'
    :param gt_channel: only apply if evaluate is True. the ground truth channel of the rescaled_data
    :return: prediction masks; or prediction masks, recall, precision, dice
    """
    if scan_name[-1] == 'z':
        rescaled_array = np.load(scan_dict + scan_name)['array']
    else:
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


def one_way_prediction(rescaled_array, evaluate_sub_model=False, batch_size=64, direction='Z'):
    """
    Should be no ground truth channel!
    :param batch_size: the batch_size when predicting.
    :param evaluate_sub_model: whether evaluate the performance of single direction.
    :return: prediction masks; or prediction masks, recall, precision, dice
    """

    if direction == 'X':
        check_point_path_X = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/check_points/hayida/stage_one/Z/0_saved_model.pth'
        return test_model.predict_one_scan_binary_class(rescaled_array, 'X', check_point_path_X, array_info,
                                                                evaluate=evaluate_sub_model, batch_size=batch_size)
    elif direction == 'Y':
        check_point_path_Y = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/check_points/hayida/stage_one/Z/0_saved_model.pth'
        return test_model.predict_one_scan_binary_class(rescaled_array, 'Y', check_point_path_Y, array_info,
                                                                evaluate=evaluate_sub_model, batch_size=batch_size)
    elif direction == 'Z':
        check_point_path_Z = '/home/zhoul0a/Desktop/Breast_Cancer_MICCAI/new/check_points/hayida/stage_one/Z/0_saved_model.pth'
        return test_model.predict_one_scan_binary_class(rescaled_array, 'Z', check_point_path_Z, array_info,
                                                                evaluate=evaluate_sub_model, batch_size=batch_size)
    else:
        print("wrong input")
        return None


def get_enhance_channel(scan_dict, scan_name, lung_mask_dict, check_point_dict_semantic, ratio_low, ratio_high,
                        batch_size=64, lung_mask=None):
    """
    :param lung_mask: you can give lung mask as npy float32 array.
    :param scan_dict: the directory where the CT data stores. should in shape [512, 512, 512] or [512, 512, 512, 2] and
    [:, :, :, 0] is the CT data.
    :param scan_name: like xwqg-A00121_2019-05-15.npy
    :param lung_mask_dict: where the lung_masks are saved, in .npz format
    :param check_point_dict_semantic: there are three folder: X/, Y/, Z/, each with 5 files named like 4_saved_model.pth
    ratio, is defined as: volume_semantic / volume lung
    :param ratio_low: a float like 0.043, which means we want to reach high precision, only take 0.043*np.sum(mask_lung)
    as predicted positive.
    :param ratio_high: a float like 0.108, which means we want to reach high recall, take up to 0.108*np.sum(mask_lung)
    as predicted positive
    :param batch_size: the batch_size for 2D U-nets
    :return: two arrays both with shape [512, 512, 512]. one is the high recall mask, the other is high precision mask
    and the rescaled_array
    """
    if scan_name[-1] == 'y':
        rescaled_array = np.load(scan_dict + scan_name)
    else:
        rescaled_array = np.load(scan_dict + scan_name)['array']

    if lung_mask is None:
        if os.path.exists(os.path.join(lung_mask_dict, scan_name[:-4] + '.npz')):
            lung_mask_path = os.path.join(lung_mask_dict, scan_name[:-4] + '.npz')
        else:
            lung_mask_path = os.path.join(lung_mask_dict, scan_name[:-4] + '_refine.npz')

        lung_mask = np.load(lung_mask_path)['array']

    lung_pixels = np.sum(lung_mask)
    print("there are:", lung_pixels, "of lung voxels")

    inside_lung = np.where(lung_mask > 0)
    x_min = max(np.min(inside_lung[0]), 10)
    x_max = min(np.max(inside_lung[0]), 500)
    y_min = max(np.min(inside_lung[1]), 10)
    y_max = min(np.max(inside_lung[1]), 500)
    z_min = max(np.min(inside_lung[2]), 10)
    z_max = min(np.max(inside_lung[2]), 500)
    print("x_min, x_max:", x_min, x_max)
    print("y_min, y_max:", y_min, y_max)
    print("z_min, z_max:", z_min, z_max)

    lung_range = (x_min, x_max, y_min, y_max, z_min, z_max)
    print("lung range is:", lung_range)

    check_point_path_X = check_point_path_generator(check_point_dict_semantic, scan_name, 'X')
    check_point_path_Y = check_point_path_generator(check_point_dict_semantic, scan_name, 'Y')
    check_point_path_Z = check_point_path_generator(check_point_dict_semantic, scan_name, 'Z')

    prediction_X = test_model.predict_one_scan_binary_class(rescaled_array, 'X', check_point_path_X, array_info,
                                                            evaluate=False, batch_size=batch_size)
    prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array, 'Y', check_point_path_Y, array_info,
                                                            evaluate=False, batch_size=batch_size)
    prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array, 'Z', check_point_path_Z, array_info,
                                                            evaluate=False, batch_size=batch_size)

    prediction_combined = prediction_X + prediction_Y + prediction_Z

    temp_array = np.array(prediction_combined[x_min: x_max, y_min: y_max, z_min: z_max], 'float32')
    temp_array = -np.reshape(temp_array, [-1, ])
    print("getting optimal threshold...")
    temp_array = -np.sort(temp_array)

    # high precision threshold:
    print("high precision threshold assume that semantic volume/lung volume =", ratio_low)
    threshold_precision = temp_array[int(lung_pixels * ratio_low)]
    print("threshold for high precision:", threshold_precision)

    # high recall threshold:
    print("high recall threshold assume that semantic volume/lung volume =", ratio_high)
    threshold_recall = temp_array[int(lung_pixels * ratio_high)]
    print("threshold for high recall:", threshold_recall)

    if len(np.shape(rescaled_array)) == 4:
        return np.array(prediction_combined > threshold_recall, 'float32'), \
               np.array(prediction_combined > threshold_precision, 'float32'), rescaled_array
    return np.array(prediction_combined > threshold_recall, 'float32'), \
           np.array(prediction_combined > threshold_precision, 'float32')


def get_enhance_channel_2(data_ct, lung_mask, check_point_dict_semantic, ratio_low, ratio_high, batch_size=64):
    """
    :param data_ct: npy array in float32 and should in shape [512, 512, 512] or [512, 512, 512, 2] and
    [:, :, :, 0] is the CT data.
    :param lung_mask: npy array in float 32 and in shape [512, 512, 512]
    :param check_point_dict_semantic: contains three models, best_model-X.pth, best_model-Y.pth, best_model-Z.pth

    ratio, is defined as: volume_semantic / volume lung
    :param ratio_low: a float like 0.043, which means we want to reach high precision, only take 0.043*np.sum(mask_lung)
    as predicted positive.
    :param ratio_high: a float like 0.108, which means we want to reach high recall, take up to 0.108*np.sum(mask_lung)
    as predicted positive
    :param batch_size: the batch_size for 2D U-nets
    :return: two arrays both with shape [512, 512, 512]. one is the high recall mask, the other is high precision mask
    and the rescaled_array
    """
    rescaled_array = data_ct

    lung_pixels = np.sum(lung_mask)
    print("there are:", lung_pixels, "of lung voxels")

    inside_lung = np.where(lung_mask > 0)
    x_min = max(np.min(inside_lung[0]), 10)
    x_max = min(np.max(inside_lung[0]), 500)
    y_min = max(np.min(inside_lung[1]), 10)
    y_max = min(np.max(inside_lung[1]), 500)
    z_min = max(np.min(inside_lung[2]), 10)
    z_max = min(np.max(inside_lung[2]), 500)
    print("x_min, x_max:", x_min, x_max)
    print("y_min, y_max:", y_min, y_max)
    print("z_min, z_max:", z_min, z_max)

    lung_range = (x_min, x_max, y_min, y_max, z_min, z_max)
    print("lung range is:", lung_range)

    check_point_path_X = os.path.join(check_point_dict_semantic, "best_model-X.pth")
    check_point_path_Y = os.path.join(check_point_dict_semantic, "best_model-Y.pth")
    check_point_path_Z = os.path.join(check_point_dict_semantic, "best_model-Z.pth")

    prediction_X = test_model.predict_one_scan_binary_class(rescaled_array, 'X', check_point_path_X, array_info,
                                                            evaluate=False, batch_size=batch_size)
    prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array, 'Y', check_point_path_Y, array_info,
                                                            evaluate=False, batch_size=batch_size)
    prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array, 'Z', check_point_path_Z, array_info,
                                                            evaluate=False, batch_size=batch_size)

    prediction_combined = prediction_X + prediction_Y + prediction_Z

    temp_array = np.array(prediction_combined[x_min: x_max, y_min: y_max, z_min: z_max], 'float32')
    temp_array = -np.reshape(temp_array, [-1, ])
    print("getting optimal threshold...")
    temp_array = -np.sort(temp_array)

    # high precision threshold:
    print("high precision threshold assume that semantic volume/lung volume =", ratio_low)
    threshold_precision = temp_array[int(lung_pixels * ratio_low)]
    print("threshold for high precision:", threshold_precision)

    # high recall threshold:
    print("high recall threshold assume that semantic volume/lung volume =", ratio_high)
    threshold_recall = temp_array[int(lung_pixels * ratio_high)]
    print("threshold for high recall:", threshold_recall)

    if len(np.shape(rescaled_array)) == 4:
        return np.array(prediction_combined > threshold_recall, 'float32'), \
               np.array(prediction_combined > threshold_precision, 'float32'), rescaled_array
    return np.array(prediction_combined > threshold_recall, 'float32'), \
           np.array(prediction_combined > threshold_precision, 'float32')


def get_array_with_enhance_channels(scan_dict, scan_name, lung_mask_dict, check_point_dict_semantic, ratio_low=0.043,
                                    ratio_high=0.108, batch_size=64,
                                    save_dict=None, training=True, lung_mask=None):
    enhance_recall, enhance_precision, rescaled_array = get_enhance_channel(scan_dict, scan_name, lung_mask_dict,
                                                                            check_point_dict_semantic, ratio_low,
                                                                            ratio_high, batch_size, lung_mask=lung_mask)
    if not training:
        # data channel + enhance_channel
        enhanced_array = np.zeros((512, 512, 512, array_info["data_channel"] + 2), 'float32')
        enhanced_array[:, :, :, 0: array_info["data_channel"]] = rescaled_array[:, :, :, 0: array_info["data_channel"]]
        enhanced_array[:, :, :, array_info["data_channel"]] = enhance_recall
        enhanced_array[:, :, :, array_info["data_channel"] + 1] = enhance_precision
    else:
        # data channel + enhance_channel + semantic_channel
        enhanced_array = np.zeros(
            (512, 512, 512, array_info["data_channel"] + len(array_info["positive_semantic_channel"]) + 2), 'float32')
        enhanced_array[:, :, :, 0: array_info["data_channel"]] = rescaled_array[:, :, :, 0: array_info["data_channel"]]
        enhanced_array[:, :, :, array_info["data_channel"]] = enhance_recall
        enhanced_array[:, :, :, array_info["data_channel"] + 1] = enhance_precision
        enhanced_array[:, :, :, array_info["data_channel"] + 2::] = rescaled_array[:, :, :,
                                                                    array_info["data_channel"]::]
    if save_dict is not None:
        Functions.save_np_array(save_dict, scan_name[:-4] + '.npz', enhanced_array, compress=True)

    return enhanced_array


def save_array_with_enhance_channels(scan_dict, lung_mask_dict, check_point_dict_semantic, ratio_low,
                                     ratio_high, batch_size,
                                     save_dict, training=True):
    scan_name_list = os.listdir(scan_dict)
    num_left = len(scan_name_list)
    for scan_name in scan_name_list:
        print("processing:", scan_name)
        print(num_left, "left")

        get_array_with_enhance_channels(scan_dict, scan_name, lung_mask_dict, check_point_dict_semantic, ratio_low,
                                        ratio_high, batch_size,
                                        save_dict, training)
        num_left -= 1


def two_stage_three_way_blood_vessel(scan_dict, scan_name, check_point_stage_one, check_point_stage_two, lung_model,
                                     ratio, ratio_low=0.043,
                                     ratio_high=0.108, batch_size=64, save_dict=None, evaluate=True, threshold=None):
    data_ct = np.load(os.path.join(scan_dict, scan_name))
    if len(np.shape(data_ct)) == 4:
        data_ct = data_ct[:, :, :, 0]
    print("get lung_mask")
    lung_mask = predict_lung(os.path.join(scan_dict, scan_name), lung_model, None, batch_size=batch_size)

    enhanced_array = get_array_with_enhance_channels(scan_dict, scan_name, None, check_point_stage_one,
                                                     lung_mask=lung_mask, training=evaluate)

    check_point_path_X = check_point_path_generator(check_point_stage_two, scan_name, 'X')
    check_point_path_Y = check_point_path_generator(check_point_stage_two, scan_name, 'Y')
    check_point_path_Z = check_point_path_generator(check_point_stage_two, scan_name, 'Z')

    old_enhance_info = array_info["enhanced_channel"]
    array_info["enhanced_channel"] = 2

    prediction_X = test_model.predict_one_scan_binary_class(enhanced_array, 'X', check_point_path_X, array_info,
                                                            evaluate=evaluate, batch_size=batch_size)
    prediction_Y = test_model.predict_one_scan_binary_class(enhanced_array, 'Y', check_point_path_Y, array_info,
                                                            evaluate=evaluate, batch_size=batch_size)
    prediction_Z = test_model.predict_one_scan_binary_class(enhanced_array, 'Z', check_point_path_Z, array_info,
                                                            evaluate=evaluate, batch_size=batch_size)
    array_info["enhanced_channel"] = old_enhance_info
    if threshold is None:
        return prediction_X + prediction_Y + prediction_Z
    else:
        prediction = np.array(prediction_X + prediction_Y + prediction_Z > threshold, 'float32')

    shape_original = np.shape(enhanced_array)
    if evaluate and shape_original[3] > 1:
        recall, precision, dice = Functions.evaluate(prediction, enhanced_array[:, :, :, 3], threshold=0)
        # if not array_info["mute_output"]:
        print("overall recall, precision, dice for", scan_name, "is", recall, precision, dice)
        return prediction, recall, precision, dice
    return prediction


# for air-way seg algorithm developed in mid July
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
                                                                   threshold=threshold, evaluate=True,
                                                                   gt_channel=gt_channel)
            recall_list.append(recall)
            precision_list.append(precision)
            dice_list.append(dice)
            print("current mean, recall, precision, dice", np.mean(recall_list), np.mean(precision_list),
                  np.mean(dice_list))
            predicted += 1
        print('\n###################\n')
        print("the performance under threshold", threshold, 'is')
        print("recall, precision, dice:", np.mean(recall_list), np.mean(precision_list), np.mean(dice_list))
        print(recall_list_threshold)
        print(precision_list_threshold)
        print(dice_list_threshold)
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


def three_way_get_enhanced_channel_and_save(rescaled_array_dict, check_point_dict, save_dict, batch_size=64):
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
        '''
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
        '''
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

        lung_mask = np.load('/ibex/scratch/projects/c2052/air_tube_seg/lung_masks/' + file_name[:-1] + 'z')['array']
        lung_pixels = np.sum(lung_mask)
        print("there are:", lung_pixels, "of lung voxels")

        inside_lung = np.where(lung_mask > 0)
        x_min = max(np.min(inside_lung[0]), 10)
        x_max = min(np.max(inside_lung[0]), 500)
        y_min = max(np.min(inside_lung[1]), 10)
        y_max = min(np.max(inside_lung[1]), 500)
        z_min = max(np.min(inside_lung[2]), 10)
        z_max = min(np.max(inside_lung[2]), 500)
        print("x_min, x_max:", x_min, x_max)
        print("y_min, y_max:", y_min, y_max)
        print("z_min, z_max:", z_min, z_max)

        tracheae_mask = prediction_X + prediction_Y + prediction_Z

        temp_array = np.array(tracheae_mask[x_min: x_max, y_min: y_max, z_min: z_max], 'float32')
        temp_array = -np.reshape(temp_array, [-1, ])
        print("getting optimal threshold...")
        temp_array = -np.sort(temp_array)

        threshold_1 = temp_array[int(lung_pixels * 0.0065)]  # high precision: percentile smallest 10
        threshold_2 = temp_array[int(lung_pixels * 0.018)]  # high recall: percentile biggest 90
        if threshold_2 < 2:  # this means the tracheae is really too small
            gap = 2 - threshold_2
            threshold_2 = 2.2
            threshold_1 += gap
            if threshold_1 > 2.998:
                threshold_1 = 2.998
        if threshold_1 > 2.998:
            threshold_1 = 2.998  # this means the tracheae is really too big

        high_precision_channel = np.array(tracheae_mask > threshold_1, 'float32')  # high precision
        high_recall_channel = np.array(tracheae_mask > threshold_2, 'float32')  # high recall
        print("high recall threshold is:", threshold_2)
        print("recall, precision, dice:", Functions.evaluate(high_recall_channel, rescaled_array[:, :, :, 1]))
        print("high precision threshold is:", threshold_1)
        print("recall, precision, dice:", Functions.evaluate(high_precision_channel, rescaled_array[:, :, :, 1]))

        enhanced_rescaled_array = np.zeros((old_shape[0], old_shape[1], old_shape[2], old_shape[3] + 2), 'float32')
        enhanced_rescaled_array[:, :, :, 0] = rescaled_array[:, :, :, 0]  # data channel
        enhanced_rescaled_array[:, :, :, 1] = high_recall_channel
        enhanced_rescaled_array[:, :, :, 2] = high_precision_channel
        enhanced_rescaled_array[:, :, :, 3] = rescaled_array[:, :, :, 1]  # semantic channel

        print("saving enhanced_rescaled_array...\n")
        Functions.save_np_array(save_dict, file_name, enhanced_rescaled_array, False)


def predict_lung(rescaled_array_dict, check_point_dict, save_dict, threshold=2., batch_size=64, gt_channel=None):
    """
    :param rescaled_array_dict: directory where arrays of standard shape and resolution stored; or path of the
    rescaled array: if it is a .npy or .npz array, it means we only predict one scan.

    :param check_point_dict: directory where lung model stored, with name: best_model-X.pth, best_model-Y.pth,
    best_model-Z.pth, and each .pth file is a dictionary with key 'state_dict' containing model.module.state_dict() or
    model.state_dict()

    :param save_dict: directory to save .npz prediction, can be None, which means do not save
    :param threshold: controls recall and precision in the three-way-prediction
    :param batch_size: batch size for the three-way-prediction
    :param gt_channel: if is None, means do not evaluate, else, rescaled_array[:, :, :, gt_channel] will be considered
    as the ground truth mask for lung and the performances will be evaluated.
    :return: if save_dict is None, return prediction_list with order of os.listdir(rescaled_array_dict), else, return None
    """
    file_name_list = []
    if os.path.isdir(rescaled_array_dict):
        file_name_list = os.listdir(rescaled_array_dict)
        num_scans = len(file_name_list)
        print("there are:", num_scans, 'scans waiting to do lung predict.')
    elif os.path.isfile(rescaled_array_dict):
        file_name = rescaled_array_dict.split('/')[-1]
        file_name_list.append(file_name)
        num_scans = len(file_name_list)
        print("predicting scan at path:", rescaled_array_dict)
        rescaled_array_dict = rescaled_array_dict[:-len(file_name)]  # now rescaled_array_dict become directory
    else:
        print("rescaled array dict error")
        return None
    num_predicted = 0
    lung_mask_list = []  # if save_dict is None, return the lung_mask_list

    old_enhance_info = array_info["enhanced_channel"]
    old_window_info = array_info["window"]
    array_info["window"] = (-5, -2, 0, 2, 5)
    array_info["enhanced_channel"] = 0

    for file_name in file_name_list:
        print("\npredicting scan", file_name, '\n', num_scans - num_predicted, 'left', ', total scans:', num_scans)
        rescaled_array = np.load(rescaled_array_dict + file_name)

        prediction_X = test_model.predict_one_scan_binary_class(rescaled_array, 'X', check_point_dict + 'best_model-X.pth',
                                                                array_info,
                                                                evaluate=False, batch_size=batch_size)
        prediction_Y = test_model.predict_one_scan_binary_class(rescaled_array, 'Y', check_point_dict + 'best_model-Y.pth',
                                                                array_info,
                                                                evaluate=False, batch_size=batch_size)
        prediction_Z = test_model.predict_one_scan_binary_class(rescaled_array, 'Z', check_point_dict + 'best_model-Z.pth',
                                                                array_info,
                                                                evaluate=False, batch_size=batch_size)
        lung_mask = np.array(prediction_X + prediction_Y + prediction_Z > threshold, 'float32')

        if save_dict is None:
            lung_mask_list.append(lung_mask)
        else:
            Functions.save_np_array(save_dict, file_name[:-4], lung_mask, True)
        if gt_channel is not None:
            ground_truth_lung_mask = rescaled_array_dict[:, :, :, gt_channel]
            print("recall, precision, dice:", Functions.evaluate(lung_mask, ground_truth_lung_mask))
        num_predicted += 1

    array_info["enhanced_channel"] = old_enhance_info
    array_info["window"] = old_window_info
    if save_dict is None and len(lung_mask_list) > 1:
        return lung_mask_list
    if save_dict is None and len(lung_mask_list) == 1:
        return lung_mask_list[0]
    return None


if __name__ == '__main__':
    exit()
    save_array_with_enhance_channels('/ibex/scratch/projects/c2052/blood_vessel_seg/arrays_raw/',
                                     '/ibex/scratch/projects/c2052/blood_vessel_seg/lung_mask/',
                                     '/ibex/scratch/projects/c2052/blood_vessel_seg/check_points/stage_one/balance_weight_array/',
                                     ratio_high=0.108, ratio_low=0.043, batch_size=64,
                                     save_dict='/ibex/scratch/projects/c2052/blood_vessel_seg/arrays_raw_with_enhanced_channel/')
    exit()
    predict_lung('/ibex/scratch/projects/c2052/blood_vessel_seg/arrays_raw/',
                 '/ibex/scratch/projects/c2052/blood_vessel_seg/check_points/model_lung/',
                 '/ibex/scratch/projects/c2052/blood_vessel_seg/lung_mask/')
    exit()
    three_way_get_enhanced_channel_and_save('/ibex/scratch/projects/c2052/air_tube_seg/arrays_raw/',
                                            '/ibex/scratch/projects/c2052/air_tube_seg/check_points/balance_weight_array/',
                                            '/ibex/scratch/projects/c2052/air_tube_seg/enhanced_two_fixed_ratio/')
    exit()

    predict_lung('/ibex/scratch/projects/c2052/air_tube_seg/arrays_raw/',
                 '/ibex/scratch/projects/c2052/air_tube_seg/check_points/model_lung/',
                 '/ibex/scratch/projects/c2052/air_tube_seg/lung_masks/', threshold=2)
    exit()
    get_optimal_threshold('/ibex/scratch/projects/c2052/air_tube_seg/arrays_raw/',
                          ['/ibex/scratch/projects/c2052/air_tube_seg/check_points/balance_weight_array/', ], 0.5,
                          three_way_prediction, search_paras=(2.3, 3.0, 0.05), gt_channel=1)

    exit()
    get_optimal_threshold('/ibex/scratch/projects/c2052/air_tube_seg/enhanced_two_channel_rescaled_array/',
                          [
                              '/ibex/scratch/projects/c2052/air_tube_seg/check_points_enhanced_two_light/balance_weight_array_/', ],
                          0.5,
                          three_way_prediction, search_paras=(1.6, 3.0, 0.1), gt_channel=1)
    exit()
    three_way_get_enhanced_channel_and_save('/ibex/scratch/projects/c2052/air_tube_seg/arrays_raw/',
                                            '/ibex/scratch/projects/c2052/air_tube_seg/check_points/balance_weight_array/',
                                            '/ibex/scratch/projects/c2052/air_tube_seg/enhanced_two_channel_rescaled_array/')
    exit()
