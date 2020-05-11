import read_in_CT
import visualize_mask_and_raw
import os
import numpy as np
import Functions
import normalize


def patient_id_list_and_scans():
    # return a dictionary, keys: patient_id, element: the scan time for this patient
    top_dict = Functions.get_father_dict() + '/patients/'
    scan_dict_for_patient = {}
    total_scans = 0
    patient_id_list = os.listdir(top_dict)
    for patient in patient_id_list:
        scan_dict_for_patient[patient] = os.listdir(top_dict + patient)
        total_scans += len(scan_dict_for_patient[patient])
        print('patient:', patient, 'has', len(scan_dict_for_patient[patient]), 'scans')

    print('\nwe have', len(patient_id_list), 'number of patients')
    print('we have', total_scans, 'number of scans')
    return scan_dict_for_patient


def patient_id_list_and_scans_visualized():
    # return a dictionary, keys: patient_id, element: the scan time for this patient
    top_dict = Functions.get_father_dict() + '/visualization/'
    scan_dict_for_patient = {}
    total_scans = 0
    patient_id_list = os.listdir(top_dict)
    for patient in patient_id_list:
        scan_dict_for_patient[patient] = os.listdir(top_dict + patient)
        total_scans += len(scan_dict_for_patient[patient])
        print('patient:', patient, 'visualized', len(scan_dict_for_patient[patient]), 'scans')

    print('\nwe visualized', len(patient_id_list), 'number of patients')
    print('we visualized', total_scans, 'number of scans')
    return scan_dict_for_patient


def get_shape_and_resolution_ww_wl(patient, scan):
    top_dict = Functions.get_father_dict() + '/patients/' + patient + '/' + scan + '/'
    dcm_list = os.listdir(top_dict)
    rows, columns, resolutions = read_in_CT.get_information_for_dcm(top_dict + dcm_list[0])
    return (rows, columns, len(dcm_list)), resolutions, Functions.wc_ww(top_dict + dcm_list[0])


def generate_intermediate_for_scan(patient, scan):
    # save and return the data array without normalization
    top_dict = Functions.get_father_dict() + '/patients/' + patient + '/' + scan + '/'
    data_array, resolution = read_in_CT.stack_dcm_files(top_dict)
    Functions.save_np_array(
        Functions.get_father_dict() + '/intermediate_arrays/' + patient + '/' + scan + '/', scan + '_data', data_array)
    print('intermediate saved')


def generate_normalized_for_scan(patient, scan):
    # save and return the normalized data array
    # it will also save the un-normalized data
    generate_intermediate_for_scan(patient, scan)
    top_dict = Functions.get_father_dict() + '/patients/' + patient + '/' + scan + '/'
    data_array, resolution = read_in_CT.stack_dcm_files(top_dict)
    shapes, resolutions, wc_ww = get_shape_and_resolution_ww_wl(patient, scan)
    data_array -= wc_ww[0]
    data_array = data_array / wc_ww[1]
    standard_array = normalize.rescale_to_standard(data_array, resolutions)
    Functions.save_np_array(
        Functions.get_father_dict() + '/standard/' + patient + '/' + scan + '/',  scan + '_data', standard_array)
    print('standard array saved')
    return standard_array


def predict_lung_mask(patient, scan):
    global threshold, batch_size
    # if lung_mask is an (512, 512, 512) np array, then it will use it to discard FP outside lung_mask.
    # warning, lung_mask must be rescaled
    # save both the normalized and un-normalized mask
    from U_net_predict import final_prediction
    checkpoint_root = "./model_lung/"
    data_path = Functions.get_father_dict() + '/standard/' + patient + '/' + scan + '/' + scan + '_data.npy'
    if os.path.exists(data_path):
        data_array = np.load(data_path)  # the data_array are normalized, with shape (512, 512, 512)
    else:
        data_array = generate_normalized_for_scan(patient, scan)

    best_model_fns = {
        'X': os.path.join(checkpoint_root, "best_model-X.pth"),
        'Y': os.path.join(checkpoint_root, "best_model-Y.pth"),
        'Z': os.path.join(checkpoint_root, "best_model-Z.pth")
    }

    prediction = final_prediction(
        data_array, best_model_fns, threshold=threshold, lung_mask=None, batch_size=batch_size)
    return prediction


def predict_one_scan(patient, scan):
    global threshold, batch_size
    # if lung_mask is an (512, 512, 512) np array, then it will use it to discard FP outside lung_mask.
    # warning, lung_mask must be rescaled
    # save both the normalized and un-normalized mask
    from U_net_predict import final_prediction
    checkpoint_root = "./model_infection/"
    data_path = Functions.get_father_dict() + '/standard/' + patient + '/' + scan + '/' + scan + '_data.npy'
    if os.path.exists(data_path):
        data_array = np.load(data_path)  # the data_array are normalized, with shape (512, 512, 512)
    else:
        data_array = generate_normalized_for_scan(patient, scan)

    best_model_fns = {
        'X': os.path.join(checkpoint_root, "best_model-X.pth"),
        'Y': os.path.join(checkpoint_root, "best_model-Y.pth"),
        'Z': os.path.join(checkpoint_root, "best_model-Z.pth")
    }

    lung_mask = predict_lung_mask(patient, scan)

    prediction = final_prediction(
        data_array, best_model_fns, threshold=threshold, lung_mask=lung_mask, batch_size=batch_size)

    shapes, resolution, ww_wc = get_shape_and_resolution_ww_wl(patient, scan)
    # now we cast the normalized prediction into un-normalized one.
    original_prediction = normalize.rescale_to_original(prediction, (334/512, 334/512, 1), resolution, shapes)
    Functions.save_np_array(
        Functions.get_father_dict() + '/standard/' + patient + '/' + scan + '/', scan + '_prediction', prediction, True)
    num_original_slices = np.shape(original_prediction)[2]

    Functions.save_np_array(Functions.get_father_dict() + '/intermediate_arrays/' +
                            patient + '/' + scan + '/', scan + '_prediction', original_prediction, True)
    return prediction


def check_format(patient, scan):
    normalized_data = \
        Functions.get_father_dict() + '/standard/' + patient + '/' + scan + '/' + scan + '_data.npy'
    normalized_prediction = \
        Functions.get_father_dict() + '/standard/' + patient + '/' + scan + '/' + scan + '_prediction.npz'
    intermediate_prediction = \
        Functions.get_father_dict() + '/intermediate_arrays/' + patient + '/' + scan + '/' + scan + '_prediction.npz'
    intermediate_data = \
        Functions.get_father_dict() + '/intermediate_arrays/' + patient + '/' + scan + '/' + scan + '_data.npy'
    if os.path.exists(normalized_data) and os.path.exists(normalized_prediction) and \
            os.path.exists(intermediate_prediction) and os.path.exists(intermediate_data):
        return True
    else:
        return False


def predict_all():
    scan_dict_for_patient = patient_id_list_and_scans()
    patients_list = list(scan_dict_for_patient.keys())
    for patient in patients_list:
        for scan in scan_dict_for_patient[patient]:
            if check_format(patient, scan):
                continue
            else:
                print('\n#############################')
                print('processing:', patient, scan)
                generate_normalized_for_scan(patient, scan)
                predict_one_scan(patient, scan)


def visualize(patient, scan, origin=False):
    print('visualizing:', patient, scan)
    top_dict = Functions.get_father_dict()
    if origin:
        p_id = patient
        s_id = scan
        gt = \
            np.load(
                top_dict + '/intermediate_arrays/' + p_id + '/' + s_id + '/' + s_id + '_prediction.npz')['array']

        array = np.load(
            top_dict + '/intermediate_arrays/' + p_id + '/' + s_id + '/' + s_id + '_data.npy')
        _, _, ww_wc = get_shape_and_resolution_ww_wl(p_id, s_id)
        array -= ww_wc[0]
        array = array / ww_wc[1]
        print('data has shape:', np.shape(array))
        visualize_mask_and_raw.visualize_mask_and_raw_array(gt, array,
                                                            top_dict + '/visualization/' + p_id + '/' + s_id + '/')
    else:
        p_id = patient
        s_id = scan
        gt = \
            np.load(
                top_dict + '/standard/' + p_id + '/' + s_id + '/' + s_id + '_prediction.npz')['array']

        array = np.load(
            top_dict + '/standard/' + p_id + '/' + s_id + '/' + s_id + '_data.npy')
        visualize_mask_and_raw.visualize_mask_and_raw_array(gt, array,
                                                            top_dict + '/visualization/' + p_id + '/' + s_id + '/')
        print('data has shape:', np.shape(array))


def visualize_all():
    scan_dict = patient_id_list_and_scans()
    visualized_dict = patient_id_list_and_scans_visualized()
    patient_id_list = list(scan_dict.keys())
    for patient in patient_id_list:
        if patient not in visualized_dict:
            for scan in scan_dict[patient]:
                visualize(patient, scan)
        else:
            scans = scan_dict[patient]
            visualized_scans = visualized_dict[patient]
            for scan in scans:
                if scan not in visualized_scans:
                    visualize(patient, scans)


threshold = 2  # this is a hyper-parameter. Higher threshold means less predicted infection. This value must < 3.
batch_size = 2  # if your GPU is very good, you may try to increase this
if __name__ == '__main__':
    visualize('7', 'time_1')
