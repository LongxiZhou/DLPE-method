import Functions
import numpy as np
import os


def slicing(raw_array, patient_id, time_point, direction, single=True):
    # raw_array has shape [512, 512, 512, 2] for [x, y, z, -],
    data = raw_array[:, :, :, 0]
    mask = raw_array[:, :, :, 1]
    # patient_id looks like "xgfy-A000012", time_point looks like "2012-02-19", direction looks like 'X'
    if single:
        save_dict = Functions.get_father_dict() + '/samples_for_2D_segmentator/single_slice/' + patient_id + '/' + time_point + '/' + direction + '/'
    else:
        save_dict = Functions.get_father_dict() + '/samples_for_2D_segmentator/five_slices/' + patient_id + '/' + time_point + '/' + direction + '/'
    print('patient_id, time_point, direction are:', patient_id, time_point, direction)

    if direction == 'X':
        for x in range(8, 504, 1):
            current_slice_mask = mask[x, :, :]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                if single:
                    sample = np.zeros([512, 512, 2], 'float32')
                    sample[:, :, 0] = data[x, :, :]
                    sample[:, :, 1] = current_slice_mask
                    Functions.save_np_array(save_dict, str(x), sample)
                else:
                    current_slice_data = data[x, :, :]
                    pre_five_data = data[x - 8, :, :]  # previous 5 mm is 8 slices on x-axis
                    pre_two_data = data[x - 3, :, :]  # previous 2 mm is 3 slices on x-axis
                    post_five_data = data[x + 8, :, :]
                    post_two_data = data[x + 3, :, :]
                    sample = np.zeros([512, 512, 6], 'float32')
                    sample[:, :, 0] = pre_five_data
                    sample[:, :, 1] = pre_two_data
                    sample[:, :, 2] = current_slice_data
                    sample[:, :, 3] = post_two_data
                    sample[:, :, 4] = post_five_data
                    sample[:, :, 5] = current_slice_mask
                    Functions.save_np_array(save_dict, str(x), sample)

    if direction == 'Y':
        for y in range(8, 504, 1):
            current_slice_mask = mask[:, y, :]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                if single:
                    sample = np.zeros([512, 512, 2], 'float32')
                    sample[:, :, 0] = data[:, y, :]
                    sample[:, :, 1] = current_slice_mask
                    Functions.save_np_array(save_dict, str(y), sample)
                else:
                    current_slice_data = data[:, y, :]
                    pre_five_data = data[:, y - 8, :]  # previous 5 mm is 8 slices on y-axis
                    pre_two_data = data[:, y - 3, :]  # previous 2 mm is 3 slices on y-axis
                    post_five_data = data[:, y + 8, :]
                    post_two_data = data[:, y + 3, :]
                    sample = np.zeros([512, 512, 6], 'float32')
                    sample[:, :, 0] = pre_five_data
                    sample[:, :, 1] = pre_two_data
                    sample[:, :, 2] = current_slice_data
                    sample[:, :, 3] = post_two_data
                    sample[:, :, 4] = post_five_data
                    sample[:, :, 5] = current_slice_mask
                    Functions.save_np_array(save_dict, str(y), sample)

    if direction == 'Z':
        for z in range(5, 507, 1):
            current_slice_mask = mask[:, :, z]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                if single:
                    sample = np.zeros([512, 512, 2], 'float32')
                    sample[:, :, 0] = data[:, :, z]
                    sample[:, :, 1] = current_slice_mask
                    Functions.save_np_array(save_dict, str(z), sample)
                else:
                    current_slice_data = data[:, :, z]
                    pre_five_data = data[:, :, z - 5]  # previous 5 mm is 5 slices on z-axis
                    pre_two_data = data[:, :, z - 2]  # previous 2 mm is 2 slices on z-axis
                    post_five_data = data[:, :, z + 5]
                    post_two_data = data[:, :, z + 2]
                    sample = np.zeros([512, 512, 6], 'float32')
                    sample[:, :, 0] = pre_five_data
                    sample[:, :, 1] = pre_two_data
                    sample[:, :, 2] = current_slice_data
                    sample[:, :, 3] = post_two_data
                    sample[:, :, 4] = post_five_data
                    sample[:, :, 5] = current_slice_mask
                    Functions.save_np_array(save_dict, str(z), sample)


def slicing_for_zhongxiao(raw_array, patient_id, time_point, aug, direction, single, base_dir):
    # raw_array has shape [512, 512, 512, 2] for [x, y, z, -],
    data = raw_array[:, :, :, 0]
    mask = raw_array[:, :, :, 1]
    # patient_id looks like "xgfy-A000012", time_point looks like "2012-02-19", direction looks like 'X'
    if single:
        save_dict = os.path.join(base_dir,'single_slice/')
    else:
        save_dict = os.path.join(base_dir,'five_slices/')
    print('patient_id, time_point, direction are:', patient_id, time_point, direction)
    
    name_prefix = patient_id + '_' + time_point + '_'+ ("original" if not aug else "aug-%s"%(aug)) + '_' + direction + '_'

    if direction == 'X':
        for x in range(8, 504, 1):
            current_slice_mask = mask[x, :, :]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                if single:
                    sample = np.zeros([512, 512, 2], 'float32')
                    sample[:, :, 0] = data[x, :, :]
                    sample[:, :, 1] = current_slice_mask
                    Functions.save_np_array(save_dict, name_prefix + str(x), sample)
                else:
                    current_slice_data = data[x, :, :]
                    pre_five_data = data[x - 8, :, :]  # previous 5 mm is 8 slices on x-axis
                    pre_two_data = data[x - 3, :, :]  # previous 2 mm is 3 slices on x-axis
                    post_five_data = data[x + 8, :, :]
                    post_two_data = data[x + 3, :, :]
                    sample = np.zeros([512, 512, 6], 'float32')
                    sample[:, :, 0] = pre_five_data
                    sample[:, :, 1] = pre_two_data
                    sample[:, :, 2] = current_slice_data
                    sample[:, :, 3] = post_two_data
                    sample[:, :, 4] = post_five_data
                    sample[:, :, 5] = current_slice_mask
                    Functions.save_np_array(save_dict, name_prefix + str(x), sample)

    if direction == 'Y':
        for y in range(8, 504, 1):
            current_slice_mask = mask[:, y, :]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                if single:
                    sample = np.zeros([512, 512, 2], 'float32')
                    sample[:, :, 0] = data[:, y, :]
                    sample[:, :, 1] = current_slice_mask
                    Functions.save_np_array(save_dict, name_prefix + str(y), sample)
                else:
                    current_slice_data = data[:, y, :]
                    pre_five_data = data[:, y - 8, :]  # previous 5 mm is 8 slices on y-axis
                    pre_two_data = data[:, y - 3, :]  # previous 2 mm is 3 slices on y-axis
                    post_five_data = data[:, y + 8, :]
                    post_two_data = data[:, y + 3, :]
                    sample = np.zeros([512, 512, 6], 'float32')
                    sample[:, :, 0] = pre_five_data
                    sample[:, :, 1] = pre_two_data
                    sample[:, :, 2] = current_slice_data
                    sample[:, :, 3] = post_two_data
                    sample[:, :, 4] = post_five_data
                    sample[:, :, 5] = current_slice_mask
                    Functions.save_np_array(save_dict, name_prefix + str(y), sample)

    if direction == 'Z':
        for z in range(5, 507, 1):
            current_slice_mask = mask[:, :, z]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                if single:
                    sample = np.zeros([512, 512, 2], 'float32')
                    sample[:, :, 0] = data[:, :, z]
                    sample[:, :, 1] = current_slice_mask
                    Functions.save_np_array(save_dict, name_prefix + str(z), sample)
                else:
                    current_slice_data = data[:, :, z]
                    pre_five_data = data[:, :, z - 5]  # previous 5 mm is 5 slices on z-axis
                    pre_two_data = data[:, :, z - 2]  # previous 2 mm is 2 slices on z-axis
                    post_five_data = data[:, :, z + 5]
                    post_two_data = data[:, :, z + 2]
                    sample = np.zeros([512, 512, 6], 'float32')
                    sample[:, :, 0] = pre_five_data
                    sample[:, :, 1] = pre_two_data
                    sample[:, :, 2] = current_slice_data
                    sample[:, :, 3] = post_two_data
                    sample[:, :, 4] = post_five_data
                    sample[:, :, 5] = current_slice_mask
                    Functions.save_np_array(save_dict, name_prefix + str(z), sample)


def slicing_three(raw_array, patient_id, time_point, direction,base_dir):
    # raw_array has shape [512, 512, 512, 2] for [x, y, z, -],
    data = raw_array[:, :, :, 0]
    mask = raw_array[:, :, :, 1]
    # patient_id looks like "xgfy-A000012", time_point looks like "2012-02-19", direction looks like 'X'
    save_dict = os.path.join('three_slice/')

    print('patient_id, time_point, direction are:', patient_id, time_point, direction)

    name_prefix = patient_id + '_' + time_point.replace('_', '-') + '_' + direction + '_'

    if direction == 'X':
        for x in range(8, 504, 1):
            current_slice_mask = mask[x, :, :]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                current_slice_data = data[x, :, :]
                pre_one_data = data[x - 2, :, :]  # previous 1 mm is 2 slices on x-axis
                post_one_data = data[x + 2, :, :]
                sample = np.zeros([512, 512, 4], 'float32')
                sample[:, :, 0] = pre_one_data
                sample[:, :, 1] = current_slice_data
                sample[:, :, 2] = post_one_data
                sample[:, :, 3] = current_slice_mask
                Functions.save_np_array(save_dict, name_prefix + str(x), sample)

    if direction == 'Y':
        for y in range(8, 504, 1):
            current_slice_mask = mask[:, y, :]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                current_slice_data = data[:, y, :]
                pre_one_data = data[:, y - 2, :]  # previous 1 mm is 2 slices on x-axis
                post_one_data = data[:, y + 2, :]
                sample = np.zeros([512, 512, 4], 'float32')
                sample[:, :, 0] = pre_one_data
                sample[:, :, 1] = current_slice_data
                sample[:, :, 2] = post_one_data
                sample[:, :, 3] = current_slice_mask
                Functions.save_np_array(save_dict, name_prefix + str(y), sample)

    if direction == 'Z':
        for z in range(8, 504, 1):
            current_slice_mask = mask[:, :, z]
            if np.sum(current_slice_mask) < 5:
                continue
            else:
                current_slice_data = data[:, :, z]
                pre_one_data = data[:, :, z - 1]  # previous 1 mm is 1 slices on x-axis
                post_one_data = data[:, :, z + 1]
                sample = np.zeros([512, 512, 4], 'float32')
                sample[:, :, 0] = pre_one_data
                sample[:, :, 1] = current_slice_data
                sample[:, :, 2] = post_one_data
                sample[:, :, 3] = current_slice_mask
                Functions.save_np_array(save_dict, name_prefix + str(z), sample)



