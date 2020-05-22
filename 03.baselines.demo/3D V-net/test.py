import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
import glob
import shutil
import argparse
import pprint
import collections
import pandas as pd

import Vnet
import dataset
import dice_loss
import slicer_and_reconstructor as snr


def test():
    model_path="/ibex/scratch/projects/c2052/my3dvnet/checkpoint_dirs/pneumonia/best_model.pth"
    arrays_dir="/ibex/scratch/projects/c2052/COVID-19/arrays_raw"
    lung_mask_dir="/ibex/scratch/projects/c2052/COVID-19/arrays_raw_lung_seg"
    prediction_dir="./prediction"

    use_before=True
    if not os.path.isdir(prediction_dir):
        os.makedirs(prediction_dir)
    batch_size=3
    threshold=0.5
    arrays_fn=os.listdir(arrays_dir)
    assert all(map(os.path.isfile,[os.path.join(arrays_dir,array_fn) for array_fn in arrays_fn]))

    print("loading model")

    model=Vnet.VNetLight(elu=False,in_channels=1,classes=2)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    data_dict=torch.load(model_path)
    model=model.to(device)
    if type(model)==nn.DataParallel:
        model.module.load_state_dict(data_dict["state_dict"])
    else:
        model.load_state_dict(data_dict["state_dict"])

    result_fp=os.path.join(prediction_dir,"result.csv")
    result_dict={
        'filename':[],
        'Precision':[],
        "Recall":[],
        "F1_score":[],
        "pred_percentage":[],
        "gt_percentage":[],
    }
    for fn in arrays_fn:
        fp=os.path.join(arrays_dir,fn)
        prediction_file=os.path.join(prediction_dir,"prediction-"+fn)
        if os.path.isfile(prediction_file) and use_before:
            print("Load from saved prediction file: %s"%(prediction_file))
            pred=np.load(prediction_file)
        else:
            print("predicting",fn)
            pred=vnet_prediction(fp,model,batch_size,threshold=threshold,filter_predictions=True,lung_mask=None)
            np.save(prediction_file,pred)
        array=np.load(fp)
        gt=array[:,:,:,1]
        lung_mask=np.load(os.path.join(lung_mask_dir,fn))[:,:,:,1]
        res=f1_score_evaluation(pred,gt, lung_mask)
        for k,v in res.items():
            result_dict[k].append(v)
        result_dict["filename"].append(fn)
    result_table=pd.DataFrame(result_dict)
    result_table.to_csv(result_fp,index=None)
def n_parameters():
    model=Vnet.VNetLight(elu=False,in_channels=1,classes=2)
    print(model.count_params())
def vnet_prediction(image_fp,model,batch_size,threshold=0.5,filter_predictions=True,lung_mask=None): 
    model.eval()
    pred_list=[]
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    for i,sample in enumerate(dataset.COVID19_3D_Dataset_Iterator([image_fp],batch_size,shuffle=False,discard=False,label=True)):
        print("\tstep %d"%(i))
        image=sample["image"].to(device)
        pred=model(image)
        pred_softmax=F.softmax(pred,dim=1)
        pred_pos_class=pred_softmax[:,1,:,:,:]
        pred_mask=pred_pos_class>threshold
        pred_numpy=pred_mask.cpu().numpy()
        pred_list+=[arr.squeeze() for arr in np.split(pred_numpy,pred_numpy.shape[0],0)]
    prediction=snr.reconstructor(pred_list,original_shape=(512,512,512))
    if filter_predictions:
        prediction=scipy.ndimage.maximum_filter(prediction,3)
    if type(lung_mask)==np.ndarray:
        prediction=prediction*lung_mask
    return prediction

def calculate_f1_score_cpu(prediction, ground_truth, strict=False):
    # this is the patient level acc function.
    # the input for prediction and ground_truth must be the same, and the shape should be [height, width, thickness]
    # the ground_truth should in range 0 and 1

    if np.max(prediction) > 1 or np.min(prediction) < 0:
        print("prediction is not probabilistic distribution")
        exit(0)

    if not strict:
        ground_truth = np.array(ground_truth > 0, 'float32')

        TP = np.sum(prediction * ground_truth)
        FN = np.sum((1 - prediction) * ground_truth)
        FP = np.sum(prediction) - TP
    else:
        difference = prediction - ground_truth

        TP = np.sum(prediction) - np.sum(np.array(difference > 0, 'float32') * difference)
        FN = np.sum(np.array(-difference > 0, 'float32') * (-difference))
        FP = np.sum(np.array(difference > 0, 'float32') * difference)
    eps=1e-6
    F1_score = (2*TP+eps)/(FN+FP+2*TP+eps)
    Precision=(TP+eps)/(TP+FP+eps)
    Recall=(TP+eps)/(TP+FN+eps)
    return Precision, Recall, F1_score

def f1_score_evaluation(pred,gt, lung_mask, filter_ground_truth=True):  # 2.97 is tested as close to opitmal on test

    if filter_ground_truth:
        gt=scipy.ndimage.maximum_filter(gt,3)

    Precision, Recall, F1_score = calculate_f1_score_cpu(pred, gt)
    print('\tcombined: Precision, Recall, F1_score', Precision, Recall, F1_score)
    eps=1e-6
    pred_percentage=np.sum(pred*lung_mask)/(np.sum(lung_mask)+eps)
    gt_percentage=np.sum(gt*lung_mask)/(np.sum(lung_mask)+eps)
    result = dict(Precision=Precision, Recall=Recall, F1_score=F1_score,
    pred_percentage=pred_percentage,gt_percentage=gt_percentage)
    for k,v in result.items():
        if np.isnan(v) or np.isinf(v):
            result[k]=0
    return result

if __name__=="__main__":
    n_parameters()