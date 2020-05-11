import os
import numpy as np

top_dict = '/ibex/scratch/projects/c2052/COVID-19_lizhongxiao/datasets/five_slices_rescaled/'
target_dict = '/ibex/scratch/projects/c2052/COVID-19/2D_Model/datasets/train_dir_0.125/'

slice_name = os.listdir(top_dict)

exist_name = os.listdir(target_dict)

count = 0
for sample_fn in slice_name:

    if int(sample_fn[11]) % 8 == 1 and sample_fn[33] == 'Z':
        count += 1
        if count % 10 == 0:
            print(count, sample_fn)
        sample = np.load(top_dict + sample_fn)
        single_slice = np.zeros([512, 512, 2], 'float32')
        single_slice[:, :, 0] = sample[:, :, 2]
        single_slice[:, :, 1] = sample[:, :, 5]
        np.save(target_dict + sample_fn, single_slice)
