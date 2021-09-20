import post_processing.parenchyma_enhancement as enhancement
import prediction.predict_rescaled as predictor
import Format_convert.dcm_np_converter as normalize
import Interface.visualize_example as visualize
import Tool_Functions.Functions as Functions
import torch

"""
each chest CT needs about one minute to be enhanced on one V100 GPU.
"""

trained_model_top_dict = '/home/zhoul0a/Desktop/outside_share/trained_models/'
# the directory where the trained models are saved

dcm_directory = '/home/zhoul0a/Desktop/outside_share/example_data/COVID-19 inpatient/'
# the directory where the dicom or dcm files for ONE chest CT scan

enhance_array_output_directory = '/home/zhoul0a/Desktop/outside_share/example_output/'
# the directory to save the enhanced ct data

predictor.top_directory_check_point = trained_model_top_dict

rescaled_ct_array = normalize.dcm_to_spatial_signal_rescaled(dcm_directory, wc_ww=(-600, 1600))
# you can set wc_ww to None if dcm or dicom contains information about correct lung window.

lung_mask = predictor.predict_lung_masks_rescaled_array(rescaled_ct_array, refine=False)

airway_mask = predictor.get_prediction_airway(rescaled_ct_array, lung_mask=lung_mask, semantic_ratio=0.02,
                                              refine_airway=False)
# refine_airway will leave one connected component (discard around 1% predicted positives) and cost about 30 sec.

blood_vessel_mask = predictor.get_prediction_blood_vessel(rescaled_ct_array, lung_mask=lung_mask, semantic_ratio=0.1,
                                                          refine_blood_vessel=False)
# refine_blood_vessel will leave one connected component (discard around 2% predicted positives) and cost about 60 sec.

DLPE_enhanced, w_l, w_w = enhancement.remove_airway_and_blood_vessel_general_sampling(rescaled_ct_array, lung_mask, airway_mask,
                                                                            blood_vessel_mask, window=True)
w_l = round(w_l)  # the optimal window level for observing sub-visual parenchyma lesions
w_w = round(w_w)  # the window width for observing sub-visual parenchyma lesions
print("\n\n#############################################")
print("the scan level optimal window level is:", w_l, "(HU)")
print("recommend window width is:", w_w, "(HU)")
print("#############################################\n\n")

example_slice = visualize.generate_slice(rescaled_ct_array, DLPE_enhanced, airway_mask, blood_vessel_mask,
                                         slice_z=250, show=True)

Functions.image_save(example_slice, enhance_array_output_directory + 'slice image name.png', high_resolution=True)

Functions.save_np_array(enhance_array_output_directory, 'enhanced_ct_name', DLPE_enhanced, compress=True)
