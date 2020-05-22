import numpy as np
import scipy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import glob
import shutil

import slicer_and_reconstructor as snr
class COVID19_3D_Dataset(object):
    def __init__(self,fp_list,batch_size,shuffle=False):
        self.fp_list=fp_list
        self.shuffle=shuffle
    def __len__(self):
        return len(self.fp_list)
    def __iter__(self):
        self.idx=np.arange(len(self))
        if self.shuffle:
            np.random.shuffle(self.idx)
        self.current=0
        return self
    def __next__(self):
        if self.current>=len(self):
            raise StopIteration
        else:
            array=np.load(self.fp_list[self.current])

def COVID19_3D_Dataset_Iterator(fp_list,batch_size,shuffle=False,discard=False,label=True):
    indices=np.arange(len(fp_list))
    assert all(map(os.path.isfile,fp_list))
    if shuffle:
        np.random.shuffle(indices)
    if label:
        images_list=list()
        labels_list=list()
        weights_list=list()
    else:
        images_list=list()
    for i,idx in enumerate(indices):
        array=np.load(fp_list[idx])
        array_list,weight_list=snr.slice_3d_into_sub_arrays(array, size_of_sample=(128, 128, 128), stride=(128, 128, 128), neglect_all_negative=discard)
        for j in range(0,len(array_list)):
            if label:
                images_list.append(array_list[j][None,:,:,:,0])
                labels_list.append(array_list[j][:,:,:,1])
                weights_list.append(weight_list[j])
                
                if len(images_list)>=batch_size:
                    images=np.stack(images_list,0)
                    labels=np.stack(labels_list,0)
                    weights=np.stack([np.ones((batch_size,)),np.array(weights_list)],1)
                    images_list=list()
                    labels_list=list()
                    weights_list=list()
                    yield dict(image=torch.from_numpy(images),label=torch.from_numpy(labels),weight=torch.from_numpy(weights))
            else:
                images_list.append(array_list[j][None,:,:,:])
                if len(images_list)>=batch_size:
                    images=np.stack(images_list,0)
                    images_list=list()
                    yield dict(image=torch.from_numpy(images))

        # yield remaining
        if len(images_list)>0:
            if label:
                images=np.stack(images_list,0)
                labels=np.stack(labels_list,0)
                weights=np.stack([np.ones((len(images_list),)),np.array(weights_list)],1)
                images_list=list()
                labels_list=list()
                weights_list=list()
                yield dict(image=torch.from_numpy(images),label=torch.from_numpy(labels),weight=torch.from_numpy(weights))
            else:
                images=np.stack(images_list,0)
                images_list=list()
                yield dict(image=torch.from_numpy(images))

