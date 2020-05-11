import os
import shutil
import h5py
import numpy as np
from PIL import Image

import Functions

path = Functions.get_current_dict() + '/h5files/'
output_path = Functions.get_current_dict() + '/prediction_npz/'
datalist = os.listdir(path)
for i in datalist:
    print(i[24:27])
    if not i[24:27] == 'pre':
        continue
    print('processing:', i)
    file = h5py.File(path + i,'r')
    ar = file['predictions'][1,:,:,:]
    softmax_ar = np.exp(ar) / np.sum(np.exp(file['predictions']), axis=0)
    softmax_ar = np.array(softmax_ar, 'float32')
    np.savez(output_path+i[0:-4]+'.npz',prediction = softmax_ar)

