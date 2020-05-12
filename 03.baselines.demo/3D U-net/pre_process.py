import os
import shutil
import h5py
import numpy as np
from PIL import Image
import Functions

main_data_path = Functions.get_current_dict() + '/arrays_raw/'
destin_path=Functions.get_current_dict() + '/h5files/'

test_patient_list =  os.listdir(main_data_path)

for i in test_patient_list:

	sub = np.load(main_data_path + i)
	sub_raw_data = sub[:,:,:]
	sub_label = sub[:,:,:] # there is no label when testing
	train= h5py.File(destin_path+i[0:-4]+'.h5','w')
	train.create_dataset('raw',data = sub_raw_data)
	train.create_dataset('label',data = sub_label)
	train.close()
