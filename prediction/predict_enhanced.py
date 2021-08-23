from post_processing.parenchyma_enhancement import remove_airway_and_blood_vessel_general_sampling
from post_processing.parenchyma_enhancement import remove_airway_and_blood_vessel_based_on_upper_frontal
import prediction.predict_rescaled as predictor
import numpy as np
import os
import prediction.three_way_prediction as three_way_prediction
top_directory_check_point = '/trained_models/'
# where the checkpoints stored: top_directory_check_point/model_type/direction/best_model-direction.pth
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # use two V100 GPU

array_info = {
    "resolution": (1, 1, 1),
    "data_channel": 1,
    # infection and lung, and stage one data_channel is 1; stage two for tracheae, airways vessel is 1
    "enhanced_channel": 2,
    # infection and lung, stage one, enhance_channel is 0; stage two for tracheae, airways vessel
    # is 2
    "window": (-1, 0, 1),  # infection, lung, window is (-5, -2, 0, 2, 5); tracheae and airways vessel (-1, 0, 1)
    "positive_semantic_channel": None,  # prediction phase this should be None
    "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
    "mute_output": True,  # if you want to see prediction details, set is as False
    "wrong_scan": None,
    "init_features": 16
}


def get_prediction_rescaled_array(rescaled_array, check_point_dict, threshold=None, batch_size=64):
    """
    :param rescaled_array: numpy array in shape [512, 512, 512]
    :param check_point_dict: where the model saved, should in check_point_dict/direction/model_name.pth
    :param threshold: the threshold for the three way prediction, should in (0, 3), or None. None means return the sum
    of the probability map of three directions.
    :param batch_size: the batch_size when prediction
    :return: the prediction
    """
    assert np.shape(rescaled_array) == (512, 512, 512)
    assert len(os.listdir(check_point_dict)) == 3
    for direction in os.listdir(check_point_dict):
        assert len(os.listdir(os.path.join(check_point_dict, direction))) == 1
    print("check_point_dict:", check_point_dict)
    prediction = three_way_prediction.three_way_predict_binary_class(rescaled_array, check_point_dict, array_info, threshold,
                                                                     batch_size)
    return prediction


def get_prediction_invisible_lesion_covid_19(rescaled_array, check_point_top_dict=None, lung_mask=None, airway=None,
                                             blood_vessel=None, threshold=2., batch_size=64, general_sampling=False, save_enhance_path=None):
    # 0 < threshold < 3, or threshold == None to return the sum of probability maps.
    global array_info
    original_array_info = array_info
    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    if lung_mask is None:
        lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict, batch_size, False)
    if airway is None:
        airway = predictor.get_prediction_airway(rescaled_array, None, lung_mask, check_point_top_dict, batch_size)
    if blood_vessel is None:
        blood_vessel = predictor.get_prediction_blood_vessel(rescaled_array, None, lung_mask, check_point_top_dict, batch_size)
    print("get enhanced_array")
    if not general_sampling:
        enhanced_array = remove_airway_and_blood_vessel_based_on_upper_frontal(rescaled_array, lung_mask, airway,
                                                                               blood_vessel)
    else:
        enhanced_array = remove_airway_and_blood_vessel_general_sampling(rescaled_array, lung_mask, airway,
                                                                         blood_vessel)
    if save_enhance_path is not None:
        import Tool_Functions.Functions as Functions
        file_name = save_enhance_path.split('/')[-1]
        length_file_name = len(file_name)
        if save_enhance_path[-1] == 'z':
            Functions.save_np_array(save_enhance_path[0:-length_file_name], file_name, enhanced_array, True)
        else:
            Functions.save_np_array(save_enhance_path[0:-length_file_name], file_name, enhanced_array, False)
    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }

    print("predicting invisible infection for covid-19 patients\n")
    check_point_directory = os.path.join(check_point_top_dict, 'invisible_COVID-19_lesion/')
    infection_mask = get_prediction_rescaled_array(enhanced_array, check_point_directory, threshold, batch_size)
    infection_mask = infection_mask * lung_mask
    infection_mask = infection_mask * (1 - blood_vessel)
    infection_mask = infection_mask * (1 - airway)
    array_info = original_array_info
    return infection_mask


def get_invisible_covid_19_lesion_from_enhanced(enhanced_array, check_point_top_dict=None, rescaled_array=None,
                                                lung_mask=None, airway=None, blood_vessel=None, threshold=2., batch_size=64, follow_up=True):
    # 0 < threshold < 3, or threshold == None to return the sum of probability maps.
    global array_info
    original_array_info = array_info

    if check_point_top_dict is None:
        check_point_top_dict = top_directory_check_point
    if lung_mask is None and rescaled_array is not None:
        lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_array, check_point_top_dict, batch_size, False)
    if airway is None and rescaled_array is not None:
        airway = predictor.get_prediction_airway(rescaled_array, check_point_top_dict=check_point_top_dict,
                                                 batch_size=batch_size, lung_mask=lung_mask)
    if blood_vessel is None and rescaled_array is not None:
        blood_vessel = predictor.get_prediction_blood_vessel(rescaled_array, check_point_top_dict=check_point_top_dict,
                                                             batch_size=batch_size, lung_mask=lung_mask)

    array_info = {
        "resolution": (1, 1, 1),
        "data_channel": 1,
        "enhanced_channel": 0,
        "window": (-1, 0, 1),
        "positive_semantic_channel": None,  # prediction phase this should be None
        "output_channels": 2,  # infection, lung, tracheae, airways vessel, output_channels is 2: positive and negative
        "mute_output": True,  # if you want to see prediction details, set is as False
        "wrong_scan": None,
        "init_features": 16
    }

    print("predicting invisible infection for covid-19 patients\n")
    if follow_up:
        check_point_directory = os.path.join(check_point_top_dict, 'invisible_COVID-19_lesion_follow_up_only/')
    else:
        check_point_directory = os.path.join(check_point_top_dict, 'invisible_COVID-19_lesion_final/')
    infection_mask = get_prediction_rescaled_array(enhanced_array, check_point_directory, threshold, batch_size)
    if lung_mask is not None:
        infection_mask = infection_mask * lung_mask
    if airway is not None:
        infection_mask = infection_mask * (1 - airway)
    if blood_vessel is not None:
        infection_mask = infection_mask * (1 - blood_vessel)
    array_info = original_array_info
    return infection_mask


if __name__ == '__main__':
    import Tool_Functions.Functions as Functions
    enhanced_root_dict = '/home/zhoul0a/Desktop/prognosis_project/original_follow_up/parenchyma_enhanced_arrays/'
    fn_list = os.listdir(enhanced_root_dict)
    for fn in fn_list[0::2]:
        print('processing:', fn)
        array = np.load(enhanced_root_dict + fn)
        lung_mask = np.load('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks_refined/lung_masks/'+ fn.split('_')[0] + '/' + fn[:-4] + '_mask_refine.npz')['array']
        invisible = get_invisible_covid_19_lesion_from_enhanced(array, threshold=None)
        invisible = lung_mask * invisible
        Functions.save_np_array('/home/zhoul0a/Desktop/prognosis_project/original_follow_up/rescaled_masks/invisible_COVID-19_probability/', fn[:-4] + '_mask.npy', invisible, False)
