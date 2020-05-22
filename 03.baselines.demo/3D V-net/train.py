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

import Vnet
import dataset
import dice_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Run Training.")
    parser.add_argument('--train_data_dir', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    return parser.parse_args()
args=parse_args()
print(args)

params={
    "n_epochs":300,
    # TODO
    "batch_size":16,
    "lr":1e-3,
    # TODO
    "train_data_dir":args.train_data_dir,
    "test_data_dir":args.test_data_dir,
    "checkpoint_dir":args.checkpoint_dir,
    "saved_model_filename":"saved_model.pth",
    "best_model_filename":"best_model.pth",
    "device":"cuda:0" if torch.cuda.is_available() else "cpu"
}

def save_checkpoint(epoch,model,optimizer,is_best,best_f1,history, filename= params["saved_model_filename"]):
    filename=os.path.join(params["checkpoint_dir"],filename)
    torch.save({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if type(model)==nn.DataParallel else model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_f1':best_f1,
            'history':history
        }, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(params["checkpoint_dir"],params["best_model_filename"]))

def weighted_binary_3d_cross_entropy(pred,gt,weights):
    val=-F.log_softmax(pred,dim=1)
    val=val*weights[:,:,None,None,None]
    val[:,0,:,:,:]=val[:,0,:,:,:]*(1-gt)
    val[:,1,:,:,:]=val[:,1,:,:,:]*gt
    return val.sum()

def train_loop_one(train_fps,params=params):
    import overfit_test
    model=Vnet.VNetLight(elu=False,in_channels=1,classes=2).cuda()
    # model=overfit_test.UNet(n_classes=2).cuda()

    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(0,100):
        print("Training epoch %d"%(epoch))
        model.train()
        train_iter=dataset.COVID19_3D_Dataset_Iterator(train_fps,params["batch_size"],shuffle=True,discard=True,label=True)
        for i,sample in enumerate(train_iter):
            while True:
                model.train()
                current_batch_size,_,width,height,depth=sample["image"].shape
                image=sample["image"].to(params["device"]).float()
                label=sample["label"].to(params["device"]).float()
                weight=sample["weight"].to(params["device"]).float()
                pred=model(image)


                loss=weighted_binary_3d_cross_entropy(pred,label,weight)/(current_batch_size*width*height*depth)
                # loss,_=dice_loss_fn(pred,label.squeeze())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("loss=%.4f"%(loss))
                
                model.eval()
                pred=model(image)
                loss=weighted_binary_3d_cross_entropy(pred,label,weight).item()
                vals=dict()
                vals["tot_pixels"]=(current_batch_size*width*height*depth)
                vals["loss"]=loss
                pred=(pred[:,1,:,:,:]>pred[:,0,:,:,:]).float().unsqueeze(1)
                pred_tp=pred*label
                if not( (label<=1).all() and (label>=0).all() ):
                    import ipdb; ipdb.set_trace()
                tp=pred_tp.sum().float().item()
                vals["tp"]=tp
                vals["fp"]=pred.sum().float().item()-tp
                if vals["fp"]<0:
                    import ipdb; ipdb.set_trace()
                pred_fn=(1-pred)*label
                vals["fn"]=pred_fn.sum().float().item()
                pred_tn=(1-pred)*(1-label)
                if (pred_tn>1).any():
                    import ipdb; ipdb.set_trace()
                vals["tn"]=pred_tn.sum().float().item()

                eps=1e-6
                vals["loss"]=vals["loss"]/vals["tot_pixels"]
                vals["precision"]=(vals["tp"]+eps)/(vals["tp"]+vals["fp"]+eps)
                vals["recall"]=(vals["tp"]+eps)/(vals["tp"]+vals["fn"]+eps)
                vals["f1"]=2* (vals["precision"]*vals["recall"]+eps) / (vals["precision"] + vals["recall"] + eps)
                print(vals)

def train_loop_random_data(params):
    import overfit_test
    print("random data test")
    loss_fn=nn.CrossEntropyLoss(reduction="sum")
    model=Vnet.VNetLight(elu=False,in_channels=1,classes=2).cuda()
    # model=overfit_test.UNet(n_classes=2).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    image = torch.randn(4, 1, 64, 64, 64).cuda()
    label = torch.randint(0, 2, (4, 64, 64, 64)).float().cuda()
    weight=torch.FloatTensor([[1,1]]).cuda()
    count=0
    while True:
        count+=1
        print(count)
        model.train()
        
        current_batch_size=1
        current_batch_size,width,height,depth=label.shape

        pred=model(image)
        loss=weighted_binary_3d_cross_entropy(pred,label,weight)/(current_batch_size*width*height*depth)
        # loss=loss_fn(pred,label)/(current_batch_size*width*height*depth)
        # loss,_=dice_loss_fn(pred,label.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss=%.4f"%(loss))
        
        model.eval()
        pred=model(image)
        loss=weighted_binary_3d_cross_entropy(pred,label,weight)
        vals=dict()
        vals["tot_pixels"]=(current_batch_size*width*height*depth)
        vals["loss"]=loss
        pred=(pred[:,1,:,:,:]>pred[:,0,:,:,:]).float()
        pred_tp=pred*label
        tp=pred_tp.sum().float().item()
        vals["tp"]=tp
        vals["fp"]=pred.sum().float().item()-tp
        pred_fn=(1-pred)*label
        vals["fn"]=pred_fn.sum().float().item()
        pred_tn=(1-pred)*(1-label)
        vals["tn"]=pred_tn.sum().float().item()

        eps=1e-6
        vals["loss"]=vals["loss"]/vals["tot_pixels"]
        vals["precision"]=(vals["tp"]+eps)/(vals["tp"]+vals["fp"]+eps)
        vals["recall"]=(vals["tp"]+eps)/(vals["tp"]+vals["fn"]+eps)
        vals["f1"]=2* (vals["precision"]*vals["recall"]+eps) / (vals["precision"] + vals["recall"] + eps)
        print(vals)                

def train_loop(model,optimizer,train_fps,test_fps,params=params,resume=True,evaluate_best=True):
    dice_loss_fn=dice_loss.DiceLoss(2,weight=torch.tensor([1,1]).cuda())
    saved_model_path=os.path.join(params["checkpoint_dir"],params["saved_model_filename"])
    if resume and os.path.isfile(saved_model_path):
        data_dict=torch.load(saved_model_path)
        epoch_start=data_dict["epoch"]
        if type(model)==nn.DataParallel:
            model.module.load_state_dict(data_dict["state_dict"])
        else:
            model.load_state_dict(data_dict["state_dict"])
        optimizer.load_state_dict(data_dict["optimizer"])
        best_f1=data_dict["best_f1"]
        history=data_dict["history"]
    else:
        epoch_start=0
        best_f1=-float('inf')
        history=collections.defaultdict(list)
    print("Going to train epochs [%d-%d]"%(epoch_start+1,epoch_start+params["n_epochs"]))

    for epoch in range(epoch_start+1,epoch_start+1+params["n_epochs"]):
        print("Training epoch %d"%(epoch))
        model.train()
        train_iter=dataset.COVID19_3D_Dataset_Iterator(train_fps,params["batch_size"],shuffle=True,discard=True,label=True)
        for i,sample in enumerate(train_iter):
            current_batch_size,_,width,height,depth=sample["image"].shape
            image=sample["image"].to(params["device"]).float()
            label=sample["label"].to(params["device"]).float()
            weight=sample["weight"].to(params["device"]).float()
            pred=model(image)
            loss=weighted_binary_3d_cross_entropy(pred,label,weight)/(current_batch_size*width*height*depth)
            # loss,_=dice_loss_fn(pred,label.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("\tstep [%d], loss=%.4f"%(i+1,loss))
        print("\tEvaluating")


        # TODO
        test_iter=dataset.COVID19_3D_Dataset_Iterator(train_fps,params["batch_size"],shuffle=False,discard=True,label=True)
        eval_vals_test=evaluate(model,test_iter,params) # This will change model to evaluation mode
        for k,v in eval_vals_test.items():
            history[k+"_test"].append(v)
        print("\tloss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f"
              %(eval_vals_test["loss"],eval_vals_test["precision"],eval_vals_test["recall"],eval_vals_test["f1"]))
        print(eval_vals_test)

        # TODO
        test_iter=dataset.COVID19_3D_Dataset_Iterator(test_fps,params["batch_size"],shuffle=False,discard=True,label=True)
        eval_vals_test=evaluate(model,test_iter,params) # This will change model to evaluation mode
        for k,v in eval_vals_test.items():
            history[k+"_test"].append(v)
        print("\tloss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f"
              %(eval_vals_test["loss"],eval_vals_test["precision"],eval_vals_test["recall"],eval_vals_test["f1"]))
        print(eval_vals_test)

        if eval_vals_test["f1"]>best_f1:
            best_f1=eval_vals_test["f1"]
            flag=True
        else:
            flag=False
        save_checkpoint(epoch,model,optimizer,flag,best_f1,history)
    print("Training finished")
    
    best_model_path=os.path.join(params["checkpoint_dir"],params["best_model_filename"])
    if evaluate_best and os.path.isfile(saved_model_path):
        print("Evaluating best model")
        model_new = Vnet.VNet(in_channels=1,classes=2).cuda()
        model_new=model_new.to(params["device"])

        optimizer_new=torch.optim.Adam(model_new.parameters(),lr=params["lr"])
        data_dict=torch.load(best_model_path)
        best_epoch=data_dict["epoch"]
        model_new.load_state_dict(data_dict["state_dict"])
        optimizer_new.load_state_dict(data_dict["optimizer"])
        best_f1=data_dict["best_f1"]
        test_iter=dataset.COVID19_3D_Dataset_Iterator(test_fps,params["batch_size"],shuffle=False,discard=False,label=True)
        eval_vals=evaluate(model_new,test_iter,params)
        print("\trecorded metadata: best_epoch=%d, best_f1=%.4f"%(best_epoch,best_f1))
        print("\tloss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f"
              %(eval_vals["loss"],eval_vals["precision"],eval_vals["recall"],eval_vals["f1"]))

def evaluate(model,test_iter,params):
    model.eval()
    with torch.no_grad():
        vals={
            'loss':0,
            'tp':0,
            'fp':0,
            'fn':0,
            'tn':0,
            "tot_pixels":0,
        }
        for i,sample in enumerate(test_iter):
            current_batch_size,_,width,height,depth=sample["image"].shape
            vals["tot_pixels"]+=current_batch_size*width*height*depth
            image=sample["image"].to(params["device"]).float()
            label=sample["label"].to(params["device"]).float()
            weight=sample["weight"].to(params["device"]).float()
            pred=model(image)
            loss=weighted_binary_3d_cross_entropy(pred,label,weight).item()
            vals["loss"]+=loss
            pred=(pred[:,1,:,:,:]>pred[:,0,:,:,:]).float()
            pred_tp=pred*label
            tp=pred_tp.sum().float().item()
            vals["tp"]+=tp
            vals["fp"]+=pred.sum().float().item()-tp
            pred_fn=(1-pred)*label
            vals["fn"]+=pred_fn.sum().float().item()
            pred_tn=(1-pred)*(1-label)
            if (pred_tn>1).any():
                import ipdb; ipdb.set_trace()
            vals["tn"]+=pred_tn.sum().float().item()
        eps=1e-6
        vals["loss"]=vals["loss"]/vals["tot_pixels"]
        vals["precision"]=(vals["tp"]+eps)/(vals["tp"]+vals["fp"]+eps)
        vals["recall"]=(vals["tp"]+eps)/(vals["tp"]+vals["fn"]+eps)
        vals["f1"]=2* (vals["precision"]*vals["recall"]+eps) / (vals["precision"] + vals["recall"] + eps)
        return vals

if __name__=="__main__":
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    train_fps=glob.glob(os.path.join(params["train_data_dir"],"*.npy"))
    test_fps=glob.glob(os.path.join(params["test_data_dir"],"*.npy"))

    print("train:",params["train_data_dir"],len(train_fps))
    print("test:",params["test_data_dir"],len(test_fps))

    # TODO
    model=Vnet.VNetLight(elu=False,in_channels=1,classes=2)
    # model=Vnet.VNet(in_channels=1,classes=2)

    # TODO
    use_multi_gpu=True
    if torch.cuda.device_count() > 1 and use_multi_gpu:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model=model.to(params["device"])
    # TODO
    optimizer=torch.optim.Adam(model.parameters(),lr=params["lr"])
    train_loop(model,optimizer,train_fps,test_fps,params,resume=True)