import numpy as np
import Tool_Functions.Functions as Functions
import post_processing.parenchyma_enhancement as enhance
import prediction.predict_rescaled as tissue_seg
import visualization.visualize_3d.highlight_semantics as highlight
import os

refine = True  # if true, it will take about 5 more minutes for each CT scan.

ct_array_dict = '/home/zhoul0a/Desktop/prognosis_project/rescaled_ct_compressed/'
save_dict_lungs = '/home/zhoul0a/Desktop/prognosis_project/visualize/open_access/lung_mask/'
save_dict_airways = '/home/zhoul0a/Desktop/prognosis_project/visualize/open_access/airways_mask/'
save_dict_blood_vessel = '/home/zhoul0a/Desktop/prognosis_project/visualize/open_access/blood_vessel_mask/'
save_dict_enhance = '/home/zhoul0a/Desktop/prognosis_project/visualize/open_access/enhanced_array/'
image_save_dict = '/home/zhoul0a/Desktop/prognosis_project/visualize/open_access/visualize/'

ct_name_list = os.listdir(ct_array_dict)
total = len(ct_name_list)
processed = 0

for name in ct_name_list:
    print("process", name, ',', total - processed, 'left')
    if os.path.exists(save_dict_enhance + name[:-4] + '.npz'):
        print("processed")
        processed += 1
        continue
    rescaled_array = np.load(ct_array_dict + name)['array']

    """
    Here we segment chest tissues and do the enhancement.
    """
    lungs = tissue_seg.predict_lung_masks_rescaled_array(rescaled_array)
    airways = tissue_seg.get_prediction_airway(rescaled_array, lung_mask=lungs, refine_airway=refine)
    blood_vessel = tissue_seg.get_prediction_blood_vessel(rescaled_array, lung_mask=lungs, refine_blood_vessel=refine)
    visible_lesion = tissue_seg.predict_covid_19_infection_rescaled_array(rescaled_array, lung_mask=lungs)

    """
        Here we visualize the segmentation for lung, airways and blood vessels.
        """
    semantic_array = highlight.highlight_mask(lungs, np.clip(rescaled_array + 0.5, 0, 1), 'G', False)
    semantic_array = highlight.highlight_mask(blood_vessel, semantic_array, 'R', True)
    semantic_array = highlight.highlight_mask(airways, semantic_array, 'B', True)

    mid_z = 256

    Functions.image_save(semantic_array[:, :, mid_z], image_save_dict + name[:-4], high_resolution=True)

    """
    Here we remove visible lesions in our study. Note, if you use DLPE for other lung disease like H1N1, make sure
    to remove the visible lesions, thus, DLPE can calculate scan optimal window.
    """
    rescaled_array = rescaled_array * (1 - visible_lesion)

    enhanced_array, w_l, w_w = enhance.remove_airway_and_blood_vessel_general_sampling(rescaled_array, lung_mask=lungs,
                                                                                       blood_vessel=blood_vessel,
                                                                                       airway=airways, window=True)
    w_l = round(w_l)  # the optimal window level for observing sub-visual parenchyma lesions
    w_w = round(w_w)  # the window width for observing sub-visual parenchyma lesions
    print("\n\n#############################################")
    print("the scan level optimal window level is:", w_l, "(HU)")
    print("recommend window width is:", w_w, "(HU)")
    print("#############################################\n\n")

    """
    Here we save the masks for lungs, airways and blood vessels, and save the enhanced array. Note, here the enhanced
    array do not contain visible lesions, and radiologist can identify sub-visual lesions on these enhanced arrays.
    """
    Functions.save_np_array(save_dict_lungs, name[:-4], lungs, compress=True)
    Functions.save_np_array(save_dict_airways, name[:-4], airways, compress=True)
    Functions.save_np_array(save_dict_blood_vessel, name[:-4], blood_vessel, compress=True)
    Functions.save_np_array(save_dict_enhance, name[:-4], enhanced_array, compress=True)
