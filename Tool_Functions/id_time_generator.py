import os


def return_all_tuples_for_rescaled_ct(rescaled_ct_directory):
    id_time_list = []
    patient_id_list = os.listdir(rescaled_ct_directory)
    for patient_id in patient_id_list:
        np_array_directory = os.path.join(rescaled_ct_directory, patient_id)
        scan_name_list = os.listdir(np_array_directory)
        for name in scan_name_list:
            time = name.split('_')[1][:-4]
            id_time_list.append((patient_id, time))
    return id_time_list


def return_all_tuples_for_array_files(directory_of_arrays):
    id_time_list = []
    file_name_list = os.listdir(directory_of_arrays)
    for file_name in file_name_list:
        patient_id = file_name.split('_')[0]
        if len(file_name.split('_')) > 1:
            time = file_name.split('_')[1][:-4]
        else:
            patient_id = file_name.split('_')[0][:-4]
            time = ''
        id_time_list.append((patient_id, time))
    return id_time_list


def return_all_tuples_for_original_data(original_data_directory):
    id_time_list = []
    patient_id_list = os.listdir(original_data_directory)
    for patient_id in patient_id_list:
        raw_data_directory = os.path.join(original_data_directory, patient_id)
        time_list = os.listdir(raw_data_directory)
        for time in time_list:
            id_time_list.append((patient_id, time))
    return id_time_list


def ct_id_and_register_id(ct_id=True):
    patient_id_ct_id_dict = {
        21077573: 'xghf-24',
        21392100: 'xghf-34',
        21669005: 'xghf-23',
        21669280: 'xghf-03',
        21669721: 'xghf-04',
        21669920: 'xghf-11',
        21670082: 'xghf-01',
        21670085: 'xghf-05',
        21671420: 'xghf-27',
        21673246: 'xghf-08',
        21676363: 'xghf-02',

        20181905: 'xghf-36',
        21735492: 'xghf-30',
        21742440: 'xghf-13',
        21760506: 'xghf-31',
        21760520: 'xghf-10',
        21760851: 'xghf-18',
        21761032: 'xghf-25',
        21761396: 'xghf-09',
        21761557: 'xghf-32',
        21762099: 'xghf-38',
        21762138: 'xghf-35',

        20197076: 'xghf-42',
        20341089: 'xghf-46',
        20362812: 'xghf-29',
        20391421: 'xghf-39',
        21724341: 'xghf-22',
        21729539: 'xghf-28',
        21735179: 'xghf-17',
        21761336: 'xghf-33',
        21761877: 'xghf-12',
        21762103: 'xghf-20',
        21762131: 'xghf-19',
        21762133: 'xghf-15',
        21762220: 'xghf-26',
        21762222: 'xghf-21',
        21762226: 'xghf-07',
        21762227: 'xghf-45',
        21762231: 'xghf-37',
        21762232: 'xghf-41',
        21762233: 'xghf-40',
        21762234: 'xghf-43',
        21762235: 'xghf-14',
        21762237: 'xghf-44',
        21762238: 'xghf-16',
        21762239: 'xghf-06',
    }
    register_id_list = list(patient_id_ct_id_dict.keys())
    ct_id_list = []
    for register_id in register_id_list:
        ct_id_list.append(patient_id_ct_id_dict[register_id])
    if ct_id:
        return ct_id_list
    return register_id_list


if __name__ == '__main__':
    print(ct_id_and_register_id(False))
