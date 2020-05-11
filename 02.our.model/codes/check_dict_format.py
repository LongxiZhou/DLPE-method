import os
import process_mha

current_dict = os.getcwd()
report_path = current_dict + '/problems_report'
f = open(report_path, 'w')
f.close()
patient_id_dict = current_dict + '/patients/'

patient_id_list = os.listdir(patient_id_dict)
for patient in patient_id_list:
    if not len(patient) == 12:
        print('name error at:', patient)

already_checked = open(current_dict + '/id_already_checked', 'r')
lines = already_checked.readlines()

for line in lines:
    try:
        patient_id_list.remove(line[0:12])
    except:
        print('not file name', line, 'in \'patients\'')

already_checked.close()


def check_one_patient(patient_id):
    print('checking format for', patient_id)
    dict_for_time = patient_id_dict + patient_id + '/'
    time_list = os.listdir(dict_for_time)
    print('we collected these time points:', time_list)
    state = 1
    for time in time_list:
        print('checking time:', time)
        data_dict = dict_for_time + time + '/Data/'
        if not process_mha.check_mask_and_raw(data_dict):
            print('an error occurred!')
            state = 0
        print('\n')
    return state


already_checked = open(current_dict + '/id_already_checked', 'a')
for patient in patient_id_list:
    if check_one_patient(patient):
        already_checked.write(patient + '\n')

if os.path.getsize(report_path) == 0:
    print('Congratulations, No Error Found!')
    f = open(report_path, 'a')
    f.write('Congratulations, No Error Found!\n')
    f.close()
else:
    print('there are still error!')
