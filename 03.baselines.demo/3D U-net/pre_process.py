import os
import shutil
import h5py
import numpy as np
from PIL import Image


main_data_path = './arrays_raw/'
destin_path='./pytorch3dunet/data/'

test_patient_list =  os.listdir(main_data_path)

for i in test_patient_list:
    print(i)
    train = h5py.File(destin_path + i[0:-4] + '.h5', 'w')
    sub = np.load(main_data_path + i)
    if sub.ndim == 3:
        train.create_dataset('raw', data=sub)
    if sub.ndim == 4:
        sub_raw_data = sub[:,:,:,0]
        sub_label = sub[:,:,:,1]
        train.create_dataset('raw',data = sub_raw_data)
        train.create_dataset('label',data = sub_label)
    train.close()
