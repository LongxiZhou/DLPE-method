import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import Functions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import glob
import shutil
import pprint
import scipy.stats
import scipy.ndimage
import pickle as pkl
import sys
import warnings

import U_net_Model as unm
import U_net_loss_function as unlf
from dataset import COVID19Dataset,ToTensor,RandomFlip,RandomRotate,compute_ratio

dimension=1
print('start test')
data_root="./datasets/arrays_dir"
prediction_root="./predictions"

threshold = 0.5

model_paths={
        'Z':"./checkpoint_dirs/checkpoint_dir/saved_model.pth",
}
use_before=True
mode="single_slice"


def prediction():
    from U_net_predict import final_prediction
    from U_net_predict import f1_score_evaluation


    data_fn_list=os.listdir(data_root)
    pprint.pprint(data_fn_list)
    if not os.path.isdir(prediction_root):
        os.makedirs(prediction_root)

    for data_fn in data_fn_list:
        out_filename="prediction-%s"%(data_fn)
        

        data_array=np.load(os.path.join(data_root,data_fn))
        try:
            image = data_array[:, :, :, 0]
        except:
            image = data_array[:,:,:]
        
        # lung_mask=np.ones_like(image)
        prediction_file=os.path.join(prediction_root,out_filename)

        if os.path.isfile(prediction_file) and use_before:
            print("Load from saved prediction file: %s"%(prediction_file))
            pred=np.load(prediction_file)
        else:
            print("Predicting %s"%(data_fn))
            if dimension==1:
                assert type(model_paths)==dict and len(model_paths)==1
                if not 0<threshold<1:
                    warnings.warn("threshold is not 0<threshold<1",UserWarning)
                pred=final_prediction(image,model_paths,threshold=threshold,direction=list(model_paths.keys())[0],mode=mode)
            elif dimension==3:
                assert type(model_paths)==dict and len(model_paths)==3
                pred=final_prediction(image,model_paths,threshold=threshold,mode=mode)
            # Functions.save_np_array(prediction_file, '/' + data_fn, pred, True)
            pred = np.array(pred, 'float32')
            np.save(prediction_file,pred)

if __name__=="__main__":
    prediction()
        
    