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
import pprint
import scipy.stats
import scipy.ndimage

import U_net_Model as unm

def main():
    from U_net_predict import final_prediction
    from visualize_mask_and_raw import visualize_mask_and_raw_array
    from U_net_predict import f1_score_evaluation

    data_root="./datasets/example"
    lung_mask_root="./datasets/example/"
    checkpoint_root="./checkpoint/"
    visualization_root="./prediction_visualization/"
    prediction_root="./prediction/"
    filename="example.npy"
    lung_mask_fn="example_lung-mask.npy"
    out_filename="prediction-example.npy"
    visualization=True
    threshold=2
    best_model_fns={
        'X':os.path.join(checkpoint_root,"best_model-X.pth"),
        'Y':os.path.join(checkpoint_root,"best_model-Y.pth"),
        'Z':os.path.join(checkpoint_root,"best_model-Z.pth")
    }
    visualization_dir=visualization_root+out_filename+'/'

    if filename.endswith(".npz"):
        data_array=np.load(os.path.join(data_root,filename))["example"]
    elif filename.endswith(".npy"):
        data_array=np.load(os.path.join(data_root,filename))
    else:
        assert False
    if not os.path.isdir(checkpoint_root):
        os.mkdir(checkpoint_root)
    if not os.path.isdir(visualization_root):
        os.mkdir(visualization_root)
    if not os.path.isdir(prediction_root):
        os.mkdir(prediction_root)
    
    image=data_array[:,:,:,0]
    gt_mask=data_array[:,:,:,1]
    lung_mask=np.load(os.path.join(lung_mask_root,lung_mask_fn))
    prediction_file=os.path.join(prediction_root,out_filename)

    if os.path.isfile(prediction_file):
        print("Load from saved prediction file: %s"%(prediction_file))
        pred=np.load(prediction_file)
    else:
        print("Predicting %s"%(filename))
        pred=final_prediction(image,best_model_fns,threshold=threshold,lung_mask=lung_mask)
        np.save(prediction_file,pred)

    print("Computing F1 score")
    f1_score_evaluation(pred,gt_mask, lung_mask,filter_ground_truth=True)
    if visualization:
        print("Preparing visualization")
        visualize_mask_and_raw_array(visualization_dir,image,pred,gt_mask)

if __name__=="__main__":
    main()