"""
for new dataset, we segment the lungs,heart, airways, airways vessels, to check the data quality
for chest CT, basic_tissue will in shape [512, 512, 512, 4]
four channels are: lungs, heart, airways, blood_vessel
saved in binary, compressed
"""
import numpy as np
import os
import prediction.predict_rescaled as predictor
import Tool_Functions.Functions as Functions

load_dict_rescaled_array = '/home/zhoul0a/Desktop/pulmonary nodules/rescaled_array/'
save_dict_tissue_mask = '/home/zhoul0a/Desktop/pulmonary nodules/basic_tissue/'


def segment_one_rescaled_array(rescaled_array):
    lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_array)
    airway_mask = predictor.get_prediction_airway(rescaled_array, lung_mask=lung_mask)
    blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_array, lung_mask=lung_mask)
    heart_mask = predictor.predict_heart_rescaled_array(rescaled_array)

    tissue_mask = np.zeros([512, 512, 512, 4], 'float32')
    tissue_mask[:, :, :, 0] = lung_mask
    tissue_mask[:, :, :, 1] = heart_mask
    tissue_mask[:, :, :, 2] = airway_mask
    tissue_mask[:, :, :, 3] = blood_vessel_mask

    return tissue_mask


if __name__ == '__main__':
    rescaled_array_name_list = os.listdir(load_dict_rescaled_array)
    for file_name in rescaled_array_name_list:
        if os.path.exists(os.path.join(save_dict_tissue_mask, file_name[:-4] + '.npz')):
            print(file_name, "processed")
            continue
        print("processing", file_name)
        array_rescaled = np.load(os.path.join(load_dict_rescaled_array, file_name))
        mask_rescaled = segment_one_rescaled_array(array_rescaled)
        Functions.save_np_array(save_dict_tissue_mask, file_name[:-4], mask_rescaled, compress=True)
