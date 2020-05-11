import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import glob
import shutil

class RandomFlip(object):
    def __init__(self,dict_keys=["image","label"]):
        self.dict_keys=dict_keys
    def __call__(self,sample):
        flag=np.random.rand()>0.5
        transformed=[RandomFlip.flip_or_not(sample[k],flag) for k in self.dict_keys]
        return dict(zip(self.dict_keys,transformed))
    @staticmethod
    def flip_or_not(ts,flag):
        if flag:
            return torch.flip(ts,(1,))
        else:
            return ts

class RandomRotate(object):
    def __init__(self,dict_keys=["image","label"]):
        self.dict_keys=dict_keys
    def __call__(self,sample):
        rot=np.random.randint(4)
        transformed=[torch.rot90(sample[k],rot,(1,2)) for k in self.dict_keys]
        return dict(zip(self.dict_keys,transformed))

class ToTensor(object):
    def __init__(self,dict_keys=["image","label"]):
        self.dict_keys=dict_keys
    def __call__(self,sample):
        transformed=[torch.from_numpy(sample[k].transpose([2,0,1])) 
                     for k in self.dict_keys]
        return dict(zip(self.dict_keys,transformed))

class COVID19Dataset(torch.utils.data.Dataset):
    def __init__(self,base_dir,pattern_or_list="*.npy",transform=None,filter_dict=None,channels=5):
        self.base_dir=base_dir
        if type(pattern_or_list)==str:
            files=[os.path.basename(f) for f in glob.glob(os.path.join(base_dir,pattern_or_list))]
        elif type(pattern_or_list)==list:
            files=pattern_or_list
        files2=[]
        if filter_dict:
            assert type(filter_dict)==dict
            for fn in files:
                fields=os.path.splitext(fn)[0].split('_')
                include=True
                for k in filter_dict:
                    if fields[k] not in filter_dict[k]:
                        include=False
                        break
                if include:
                    files2.append(fn)
        else:
            files2=files
        self.files=np.array(sorted(files2)).astype(np.string_)
        self.transform=transform
        self.channels=channels
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        assert 0<=idx<len(self)
        arr=np.load(os.path.join(self.base_dir,self.files[idx].decode('utf-8')))
        image=arr[:,:,:self.channels]
        label=arr[:,:,self.channels:]
        sample={"image":image,"label":label}
        if self.transform:
            return self.transform(sample)
        else:
            return sample

def compute_ratio(base_dir,pattern="*.npy",filter_dict=None):
    transform=torchvision.transforms.Compose([
        ToTensor()
    ])
    dataset=COVID19Dataset(base_dir,pattern,transform,filter_dict)
    pos_count=0
    tot_pixel_count=0
    sample=2000
    for i,idx in enumerate(np.random.randint(0,len(dataset),sample)):
        print("[%d/%d]"%(i+1,sample),end='\r')
        arr=np.load(os.path.join(base_dir,dataset.files[idx]))
        pos_count+=arr[:,:,5].sum().item()
        tot_pixel_count+=np.prod(arr[:,:,5].shape)
    return pos_count/tot_pixel_count