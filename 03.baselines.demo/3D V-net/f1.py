import os
import shutil
import h5py
import numpy as np
from PIL import Image
from sklearn import metrics
import pandas as pd
f1_list = []
rc_list = []
pred_path = '/ibex/scratch/projects/c2052/haoyang_raw/li/vnet_npz/'
gt_path = '/ibex/scratch/projects/c2052/COVID-19/arrays_raw/'
thres = 0.5
def recall_score(y_true, y_pred):
    return ((y_true==1)*(y_pred==1)).sum()/(y_true==1).sum()
for i in os.listdir(pred_path):
    print(i[0:23])
    prediction = np.load(pred_path + i)
    gt = np.load(gt_path + i[0:23] + '.npy')
    ground_truth = gt[:,:,:,1]
    prediction = prediction['prediction']
    print(np.max(prediction))
    print(np.min(prediction))
    prediction = np.array(prediction > thres, 'float32')
    locate = np.where(prediction == 1)
    for x in range(0, len(locate[0])):
        for y in locate[1]:
            for z in locate[2]:
        if(locate[0][x] < (np.shape(prediction)[0]-1) and locate[1][x] < np.shape(prediction)[1]-1 and locate[2][x] < np.shape(prediction)[2]-1):
            prediction[locate[0][x]-1,locate[1][x],locate[2][x]] = 1
            prediction[locate[0][x]+1,locate[1][x],locate[2][x]] = 1
            prediction[locate[0][x],locate[1][x]-1,locate[2][x]] = 1
            prediction[locate[0][x],locate[1][x]+1,locate[2][x]] = 1
            prediction[locate[0][x],locate[1][x],locate[2][x]-1] = 1
            prediction[locate[0][x],locate[1][x],locate[2][x]+1] = 1

                    
    ground_truth = np.array(ground_truth > thres,'float32')
    over_lap = np.sum(prediction * ground_truth)
    f1 =  2 * over_lap / (np.sum(prediction) + np.sum(ground_truth))
    f1_list.append(f1)
    print(f1)
    prediction = np.where(prediction > thres, 1,0)
    rc_list.append(recall_score(ground_truth,prediction))
	
	
print('patient_list:',os.listdir(pred_path))
print('f1_score_list:',f1_list)
print('recall_list:',rc_list)
print('Before 0!!!!')
print('f1_score:',np.mean(f1_list))
print('recall:',np.mean(rc_list))
print('min_f1:',np.min(f1_list))
f1_list = np.array(f1_list)
rc_list = np.array(rc_list)
print('f1_var:',np.var(f1_list))
print('recall_var:',np.var(rc_list))
print('min_f1:',np.nanmin(f1_list))
print('min_num:',np.sum(f1_list < 0.2))
print('After 0!!!')
f1_list[f1_list ==  0] = np.nan
print('f1:',np.nanmean(f1_list))
rc_list[rc_list == 0] = np.nan
print('recall:',np.nanmean(rc_list))
print('f1_var:',np.nanvar(f1_list))
print('recall_var:',np.nanvar(rc_list))
print('min_f1:',np.nanmin(f1_list))
print('min_num:',np.nansum(f1_list < 0.2))
print('After 0.2!!!')
f1_list[f1_list < 0.2] = np.nan
print('f1:',np.nanmean(f1_list))
rc_list[rc_list < 0.2] = np.nan
print('recall:',np.nanmean(rc_list))
print('f1_var:',np.nanvar(f1_list))
print('recall_var:',np.nanvar(rc_list))
print('min_f1:',np.nanmin(f1_list))
print('min_num:',np.nansum(f1_list < 0.2))

