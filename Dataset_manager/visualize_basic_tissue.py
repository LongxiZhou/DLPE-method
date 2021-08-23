"""
for new dataset, we segment the lungs,heart, airways, airways vessels, to check the data quality
for chest CT, basic_tissue will in shape [512, 512, 512, 4]
four channels are: lungs, heart, airways, blood_vessel
saved in binary, compressed
"""
import numpy as np
import os
import visualization.visualize_3d.highlight_semantics as highlight
import Tool_Functions.Functions as Functions


load_rescaled_array_dict = '/home/zhoul0a/Desktop/pulmonary nodules/rescaled_array/'
load_basic_tissue_mask = '/home/zhoul0a/Desktop/pulmonary nodules/basic_tissue/'
save_image_dict = '/home/zhoul0a/Desktop/pulmonary nodules/visualization/basic_tissue_check/'


def process_one_patient(file_name):
    if os.path.exists(os.path.join(save_image_dict, file_name[:-4] + '.png')):
        print(file_name, "processed")
        return None
    else:
        print("processing:", file_name)
    rescaled_array = np.load(os.path.join(load_rescaled_array_dict, file_name))
    rescaled_array = np.clip(rescaled_array, -0.5, 0.5) + 0.5
    rescaled_mask = np.load(os.path.join(load_basic_tissue_mask, file_name[:-4] + '.npz'))['array']
    highlighted = highlight.highlight_mask(rescaled_mask[:, :, :, 0], rescaled_array, channel='B',
                                           further_highlight=False)  # lung
    highlighted = highlight.highlight_mask(rescaled_mask[:, :, :, 1], highlighted, channel='Y',
                                           further_highlight=True)  # heart
    highlighted = highlight.highlight_mask(rescaled_mask[:, :, :, 3], highlighted, channel='R',
                                           further_highlight=True)  # airways vessel
    highlighted = highlight.highlight_mask(rescaled_mask[:, :, :, 2], highlighted, channel='G',
                                           further_highlight=True)  # airway
    image = highlighted[:, :, 250, :]
    Functions.image_save(image, os.path.join(save_image_dict, file_name[:-4]), dpi=300)
    return None


def visualize_dataset():
    file_list = os.listdir(load_rescaled_array_dict)
    total_num = len(file_list)
    for file_name in file_list:
        process_one_patient(file_name)
        total_num -= 1
        print(total_num, 'left')
        print('\n')


if __name__ == '__main__':
    visualize_dataset()
