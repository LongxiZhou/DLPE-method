import os
import shutil
import h5py
import numpy as np
from PIL import Image


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
	
path = '/ibex/scratch/projects/c2052/haoyang_raw/li/vnet_post/'
output_path = '/ibex/scratch/projects/c2052/haoyang_raw/li/vnet_npz/'
datalist = os.listdir(path)

for i in datalist:
    file = h5py.File(path + i,'r')
    ar = file['predictions'][0,:,:,:]
    new_ar = sigmoid(ar)
    np.savez( output_path +i[0:-4]+'.npz',prediction = new_ar)
