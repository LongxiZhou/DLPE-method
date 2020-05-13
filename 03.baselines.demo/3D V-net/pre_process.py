import os
import shutil
import h5py
import numpy as np
from PIL import Image


main_data_path = '/home/zhoul0a/Desktop/COVID-19/reports/COVID-19-repo-master/03.baselines.demo/3D V-net/arrays_raw/'
destin_path='/home/zhoul0a/Desktop/COVID-19/reports/COVID-19-repo-master/03.baselines.demo/3D V-net/h5files/'

test_patient_list =  os.listdir(main_data_path)

for i in test_patient_list:
    print(i)
    sub = np.load(main_data_path + i)
    sub_raw_data = sub[:,:,:]
    sub_label = sub[:,:,:] #  we do not have labels, use this to hold place but the model will not use it
    sub_raw_data = (sub_raw_data - np.min(sub_raw_data))/(np.max(sub_raw_data) - np.min(sub_raw_data))
    train= h5py.File(destin_path+i[0:-4]+'.h5','w')
    train.create_dataset('raw',data = sub_raw_data)
    train.create_dataset('label',data = sub_label)
    train.close()
