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
import argparse
import pprint
import collections
import warnings
import sys

import U_net_Model as unm
import U_net_loss_function as unlf
from dataset import RandomFlip,RandomRotate,ToTensor,COVID19Dataset

TRAIN_DATA_DIR="datasets/train_dir_0.125"
CHECKPOINT_DIR="checkpoint_dirs/checkpoint_dir"

params={
    "n_epochs":30,
    "batch_size":64,
    "lr":1e-4,
    "channels":1,
    'workers':32,
    "balance_weights":[1,1],
    "train_data_dir":TRAIN_DATA_DIR,
    "test_data_dir":TRAIN_DATA_DIR, # use train for test
    "checkpoint_dir":CHECKPOINT_DIR,
    "saved_model_filename":"saved_model.pth",
    "device":"cuda:0" if torch.cuda.is_available() else "cpu"
}


def save_checkpoint(epoch,model,optimizer,history, filename= params["saved_model_filename"]):
    filename=os.path.join(params["checkpoint_dir"],filename)
    torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if type(model)==nn.DataParallel else model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'history':history
        }, filename)


def train_loop(model,optimizer,train_loader,test_loader,params=params,resume=True,evaluate_best=True):
    saved_model_path=os.path.join(params["checkpoint_dir"],params["saved_model_filename"])
    if resume and os.path.isfile(saved_model_path):
        data_dict=torch.load(saved_model_path)
        epoch_start=data_dict["epoch"]
        if type(model)==nn.DataParallel:
            model.module.load_state_dict(data_dict["state_dict"])
        else:
            model.load_state_dict(data_dict["state_dict"])
        optimizer.load_state_dict(data_dict["optimizer"])
        history=data_dict["history"]
    else:
        epoch_start=0
        history=collections.defaultdict(list)
    print("Going to train epochs [%d-%d]"%(epoch_start+1,epoch_start+params["n_epochs"]))
    for epoch in range(epoch_start+1,epoch_start+1+params["n_epochs"]):
        print("Training epoch %d"%(epoch))
        model.train()
        for i,sample in enumerate(train_loader):
            current_batch_size,_,width,height=sample["image"].shape
            image=sample["image"].to(params["device"]).float()
            label=sample["label"].to(params["device"]).float()
            pred=model(image)
            loss=unlf.cross_entropy_pixel_wise_2d_binary(pred,label,balance_weights=params["balance_weights"])/(current_batch_size*width*height)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("\tstep [%d/%d], loss=%.4f"%(i+1,len(train_loader),loss))
        print("\tEvaluating")
        eval_vals_train=evaluate(model,test_loader,params)
        for k,v in eval_vals_train.items():
            history[k+"_train"].append(v)
        print("\tloss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f"
              %(eval_vals_train["loss"],eval_vals_train["precision"],eval_vals_train["recall"],eval_vals_train["f1"]))
        
        save_checkpoint(epoch,model,optimizer,history)
    print("Training finished")
    

def evaluate(model,test_loader,params):
    # TODO: change back to eval()
    model.eval()
    with torch.no_grad():
        vals={
            'loss':0,
            'tp':0,
            'fp':0,
            'fn':0,
            "tot_pixels":0,
        }
        for i,sample in enumerate(test_loader):
            current_batch_size,_,width,height=sample["image"].shape
            vals["tot_pixels"]+=current_batch_size*width*height
            image=sample["image"].to(params["device"]).float()
            label=sample["label"].to(params["device"]).float()
            pred=model(image)
            loss=unlf.cross_entropy_pixel_wise_2d_binary(pred,label,balance_weights=params["balance_weights"]).item()
            vals["loss"]+=loss
            
            pred=(pred[:,1,:,:]>pred[:,0,:,:]).float().unsqueeze(1)
            pred_tp=pred*label
            tp=pred_tp.sum().float().item()
            vals["tp"]+=tp
            vals["fp"]+=pred.sum().float().item()-tp
            pred_fn=(1-pred)*label
            vals["fn"]+=pred_fn.sum().float().item()
        eps=1e-6
        vals["loss"]=vals["loss"]/vals["tot_pixels"]
        
        vals["precision"]=(vals["tp"]+eps)/(vals["tp"]+vals["fp"]+eps)
        vals["recall"]=(vals["tp"]+eps)/(vals["tp"]+vals["fn"]+eps)
        vals["f1"]=2* (vals["precision"]*vals["recall"]+eps) / (vals["precision"] + vals["recall"] + eps)

        if vals["tp"]+vals["fp"] <10*eps or vals["tp"]+vals["fn"]<10*eps or vals["precision"] + vals["recall"] < 10*eps:
            print("Possibly incorrect precision, recall or f1 values")
        return vals


def predict(model,test_loader,params):
    model.eval()
    prediction_list=[]
    with torch.no_grad():
        for i,sample in enumerate(test_loader):
            image=sample["image"].to(params["device"]).float()
            pred=model(image)
            pred=pred.cpu().numpy()
            prediction_list.append(pred)
        predictions=np.concatenate(prediction_list,axis=0)
        return predictions


if __name__=="__main__":
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])
    # TODO: change back
    train_transform=torchvision.transforms.Compose([
        ToTensor(),
        RandomFlip(),
        RandomRotate()
    ])
    test_transform=torchvision.transforms.Compose([
        ToTensor()
    ])

    train_dataset=COVID19Dataset(
        params["train_data_dir"],
        transform=train_transform,
        channels=params["channels"]
    )

    test_dataset=COVID19Dataset(
        params["train_data_dir"],
        transform=test_transform,
        channels=params["channels"]
    )

    print("train:",params["train_data_dir"],len(train_dataset))

    # TODO: change back to True
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["workers"])
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["workers"])

    model=unm.UNet(in_channels=params["channels"],out_channels=2)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model=model.to(params["device"])
    optimizer=torch.optim.Adam(model.parameters(),lr=params["lr"])
    train_loop(model,optimizer,train_loader,test_loader,params)
