import process_mha
import Functions
import numpy as np
import read_in_CT
import os


def visualize_mask_and_raw_array(save_dir,array_ct,pred_mask,gt_mask=None,direction='Z'):
    Functions.array_stat(array_ct)
    assert np.allclose(array_ct.shape,pred_mask.shape)
    if type(gt_mask) == np.ndarray:
        assert np.allclose(array_ct.shape,gt_mask.shape)
    
    array_cut = np.clip(array_ct, -0.5, 0.5) + 0.5

    X_size,Y_size,Z_size=array_ct.shape

    if type(gt_mask) == np.ndarray:
        merge = [np.zeros([X_size, Y_size , Z_size, 3], 'float32') for _ in range(4)]
        merge[0][:, :, :, 0] = array_cut
        merge[0][:, :, :, 1] = array_cut
        merge[0][:, :, :, 2] = array_cut

        merge[1][:, :, :, 0] = array_cut + gt_mask
        merge[1][:, :, :, 1] = array_cut - pred_mask
        merge[1][:, :, :, 2] = array_cut - pred_mask

        merge[2][:, :, :, 0] = array_cut + gt_mask
        merge[2][:, :, :, 1] = array_cut - gt_mask
        merge[2][:, :, :, 2] = array_cut - gt_mask

        TP=pred_mask*gt_mask
        FP=pred_mask*(1-gt_mask)
        FN=(1-pred_mask)*gt_mask

        merge[3][:, :, :, 0] = array_cut - TP - FP
        merge[3][:, :, :, 1] = array_cut - FN - FP
        merge[3][:, :, :, 2] = array_cut - TP - FN

    else:
        merge=[np.zeros([X_size, Y_size , Z_size, 3], 'float32') for _ in range(2)]
        merge[0][:, :, :, 0] = array_cut
        merge[0][:, :, :, 1] = array_cut
        merge[0][:, :, :, 2] = array_cut

        merge[1][:, :, :, 0] = array_cut + pred_mask
        merge[1][:, :, :, 1] = array_cut - pred_mask
        merge[1][:, :, :, 2] = array_cut - pred_mask

    merge = np.clip(merge, 0, 1)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # import ipdb; ipdb.set_trace()

    if direction=="X":
        for i in range(X_size):
            Functions.image_save(np.concatenate([merge[j][i, :, :] for j in range(len(merge))],axis=1), save_dir + str(i) +".png", gray=False)
    elif direction=="Y":
        for i in range(Y_size):
            Functions.image_save(np.concatenate([merge[j][:, i, :] for j in range(len(merge))],axis=1), save_dir + str(i) +".png", gray=False)
    elif direction=="Z":
        for i in range(Z_size):
            Functions.image_save(np.concatenate([merge[j][:, :, i] for j in range(len(merge))],axis=1), save_dir + str(i) +".png", gray=False)
    else:
        assert False

def visualize_rim(save_path,array_ct,pred_X,pred_Y,pred_Z,pred_mask,gt_mask):
    Functions.array_stat(array_ct)
    assert np.allclose(array_ct.shape,pred_mask.shape)
    if type(gt_mask) == np.ndarray:
        assert np.allclose(array_ct.shape,gt_mask.shape)
    
    array_cut = np.clip(array_ct, -0.5, 0.5) + 0.5

    X_size,Y_size=array_ct.shape

    merge = [np.zeros([X_size, Y_size, 3], 'float32') for _ in range(3)]
    merge[0][:, :, 0] = array_cut
    merge[0][:, :, 1] = array_cut
    merge[0][:, :, 2] = array_cut

    pred_Y=pred_Y*(1-pred_X)
    pred_Z=pred_Z*(1-pred_Y)*(1-pred_X)

    merge[1][:, :, 0] = array_cut - pred_Y - pred_Z
    merge[1][:, :, 1] = array_cut - pred_X - pred_Z
    merge[1][:, :, 2] = array_cut - pred_X - pred_Y

    TP=pred_mask*gt_mask
    FP=pred_mask*(1-gt_mask)
    FN=(1-pred_mask)*gt_mask

    merge[2][:, :, 0] = array_cut - TP - FP
    merge[2][:, :, 1] = array_cut - FN - FP
    merge[2][:, :, 2] = array_cut - TP - FN

    merge = [np.clip(img,0,1) for img in merge]

    # import ipdb; ipdb.set_trace()
    Functions.image_save(np.concatenate(merge,axis=1), save_path, gray=False)

def visualize_one_patient(patient_id):
    print('process:', patient_id)
    mask_list, time_list = process_mha.get_mask_array(patient_id)
    array_list, _ = read_in_CT.get_ct_array(patient_id)
    num_times = len(time_list)
    for t in range(num_times):
        print('process time point:', time_list[t], 'for patient:', patient_id)
        save_dict = Functions.get_father_dict() + '/Test/' + 'images/' + patient_id + '/' + str(time_list[t]) + '/'
        visualize_mask_and_raw_array(mask_list[t], array_list[t], save_dict)


def visualize_all():
    patient_id_list = Functions.get_patient_id()
    already_exist = os.listdir(Functions.get_father_dict() + '/Test/images/')
    for patient_id in patient_id_list:
        if patient_id in already_exist:
            continue
        visualize_one_patient(patient_id)


def visualize_rescaled():
    image_dict = Functions.get_father_dict() + '/Test/check_standard/'
    rescaled_array_dict = Functions.get_father_dict() + '/arrays_raw/'

    existing_image_files = os.listdir(image_dict)
    existing_array = os.listdir(rescaled_array_dict)

    '''
    for file in existing_image_files:
        existing_array.remove(file)
        print('the file is exist:', file)
    '''

    for array in existing_array:
        print('visualizing:', array)
        save_dict = image_dict + '0401/' + array + '/'
        rescaled_array = np.load(rescaled_array_dict + array)
        visualize_mask_and_raw_array(rescaled_array[:, :, :, 1], rescaled_array[:, :, :, 0], save_dict)


#visualize_rescaled()

