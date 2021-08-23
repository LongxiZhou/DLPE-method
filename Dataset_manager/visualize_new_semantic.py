import numpy as np
import os
import visualization.visualize_3d.highlight_semantics as highlight
import Tool_Functions.Functions as Functions


load_rescaled_array_dict = '/home/zhoul0a/Desktop/pulmonary nodules/rescaled_array/'
load_new_semantic_mask = '/home/zhoul0a/Desktop/pulmonary nodules/rescaled_gt/'
save_image_dict = '/home/zhoul0a/Desktop/pulmonary nodules/visualization/new_semantic_check/'

max_images = 10


def process_one_patient_z(file_name):
    if os.path.exists(os.path.join(save_image_dict, 'z_view', file_name[:-4] + '_0' + '.png')):
        print(file_name, "z processed")
        return None
    else:
        print("processing z:", file_name)
    rescaled_array = np.load(os.path.join(load_rescaled_array_dict, file_name))
    rescaled_array = np.clip(rescaled_array, -0.5, 0.5) + 0.5
    rescaled_mask = np.load(os.path.join(load_new_semantic_mask, file_name[:-4] + '.npz'))['array']

    total_semantic_volume = np.sum(rescaled_mask) * 334 / 512 * 334 / 512 / 1000  # cubic centimeters
    assert total_semantic_volume > 0
    print('new semantic with volume:', total_semantic_volume, 'cm^3')

    highlighted = highlight.highlight_mask(rescaled_mask, rescaled_array, channel='R',
                                           further_highlight=False)

    locations = list(set(np.where(rescaled_mask > 0.5)[2]))
    locations.sort()
    num_z_slices = len(locations)
    interval = max(int(num_z_slices / max_images), 1)
    image_count = 0
    for i in range(0, len(locations), interval):
        image = highlighted[:, :, locations[i], :]
        Functions.image_save(image, os.path.join(save_image_dict, 'z_view', file_name[:-4] + '_' + str(i)), dpi=300)
        image_count += 1
        if image_count >= 10:
            break
    return None


def process_one_patient_y(file_name):
    if os.path.exists(os.path.join(save_image_dict, 'y_view', file_name[:-4] + '_0' + '.png')):
        print(file_name, "y processed")
        return None
    else:
        print("processing y:", file_name)
    rescaled_array = np.load(os.path.join(load_rescaled_array_dict, file_name))
    rescaled_array = np.clip(rescaled_array, -0.5, 0.5) + 0.5
    rescaled_mask = np.load(os.path.join(load_new_semantic_mask, file_name[:-4] + '.npz'))['array']

    total_semantic_volume = np.sum(rescaled_mask) * 334 / 512 * 334 / 512 / 1000  # cubic centimeters
    assert total_semantic_volume > 0
    print('new semantic with volume:', total_semantic_volume, 'cm^3')

    highlighted = highlight.highlight_mask(rescaled_mask, rescaled_array, channel='R',
                                           further_highlight=False)

    locations = list(set(np.where(rescaled_mask > 0.5)[1]))
    locations.sort()
    num_y_slices = len(locations)
    interval = max(int(num_y_slices / max_images), 1)
    image_count = 0

    x_shape, y_shape, z_shape = np.shape(highlighted)[0: 3]
    for i in range(0, len(locations), interval):
        image = np.zeros([x_shape, z_shape * 2, 3], 'float32')
        image[:, 0: z_shape, :, :] = highlighted[:, locations[i], :, :]
        image[:, z_shape::, :, :] = highlighted[:, locations[i] + y_shape, :, :]
        Functions.image_save(image, os.path.join(save_image_dict, 'y_view', file_name[:-4] + '_' + str(i)), dpi=300)
        image_count += 1
        if image_count >= 10:
            break
    return None


def process_one_patient_x(file_name):
    if os.path.exists(os.path.join(save_image_dict, 'x_view', file_name[:-4] + '_0' + '.png')):
        print(file_name, "x processed")
        return None
    else:
        print("processing x:", file_name)
    rescaled_array = np.load(os.path.join(load_rescaled_array_dict, file_name))
    rescaled_array = np.clip(rescaled_array, -0.5, 0.5) + 0.5
    rescaled_mask = np.load(os.path.join(load_new_semantic_mask, file_name[:-4] + '.npz'))['array']

    total_semantic_volume = np.sum(rescaled_mask) * 334 / 512 * 334 / 512 / 1000  # cubic centimeters
    assert total_semantic_volume > 0
    print('new semantic with volume:', total_semantic_volume, 'cm^3')

    highlighted = highlight.highlight_mask(rescaled_mask, rescaled_array, channel='R',
                                           further_highlight=False)

    locations = list(set(np.where(rescaled_mask > 0.5)[0]))
    locations.sort()
    num_x_slices = len(locations)
    interval = max(int(num_x_slices / max_images), 1)
    image_count = 0
    for i in range(0, len(locations), interval):
        image = highlighted[locations[i], :, :, :]
        Functions.image_save(image, os.path.join(save_image_dict, 'x_view', file_name[:-4] + '_' + str(i)), dpi=300)
        image_count += 1
        if image_count >= 10:
            break
    return None


def visualize_dataset():
    file_list = os.listdir(load_rescaled_array_dict)
    total_num = len(file_list)
    for file_name in file_list:
        process_one_patient_x(file_name)
        process_one_patient_y(file_name)
        process_one_patient_z(file_name)
        total_num -= 1
        print(total_num, 'left')
        print('\n')


if __name__ == '__main__':
    visualize_dataset()
