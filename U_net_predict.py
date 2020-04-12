import numpy as np
import U_net_Model
import torch
import torch.nn as nn
import os
import scipy.ndimage

def load_model(parameters_dict_fn):
    model=U_net_Model.UNet(in_channels=5,out_channels=2)
    model_dict = torch.load(parameters_dict_fn)["state_dict"]
    model.load_state_dict(model_dict)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    #print("loading checkpoint")
    
    #print(" checkpoint loaded ")
    return model

def arr_at_each_location(image,direction,loc):
    arr=[]
    x_size,y_size,z_size=image.shape
    if direction=="X":
        assert 0<=loc<image.shape[0]
        arr=[
            image[loc-5,:,:] if loc-5>=0 else np.zeros((y_size,z_size)),
            image[loc-2,:,:] if loc-2>=0 else np.zeros((y_size,z_size)),
            image[loc,:,:],
            image[loc+2,:,:] if loc+2<x_size else np.zeros((y_size,z_size)),
            image[loc+5,:,:] if loc+5<x_size else np.zeros((y_size,z_size))
        ]
        arr=np.stack(arr,0)[np.newaxis,:,:,:]
    elif direction=="Y":
        assert 0<=loc<image.shape[1]
        arr=[
            image[:,loc-5,:] if loc-5>=0 else np.zeros((x_size,z_size)),
            image[:,loc-2,:] if loc-2>=0 else np.zeros((x_size,z_size)),
            image[:,loc,:],
            image[:,loc+2,:] if loc+2<y_size else np.zeros((x_size,z_size)),
            image[:,loc+5,:] if loc+5<y_size else np.zeros((x_size,z_size))
        ]
        arr=np.stack(arr,0)[np.newaxis,:,:,:]
    elif direction=="Z":
        assert 0<=loc<image.shape[2]
        arr=[
            image[:,:,loc-5] if loc-5>=0 else np.zeros((x_size,y_size)),
            image[:,:,loc-2] if loc-2>=0 else np.zeros((x_size,y_size)),
            image[:,:,loc],
            image[:,:,loc+2] if loc+2<z_size else np.zeros((x_size,y_size)),
            image[:,:,loc+5] if loc+5<z_size else np.zeros((x_size,y_size)),
        ]
        arr=np.stack(arr,0)[np.newaxis,:,:,:]
    else:
        assert False
    return arr.astype(np.float32)

def get_prediction(unet, direction, image):

    # data_array should be in shape (240, channel, 464, 464)
    step = 16
    x_size,y_size,z_size=image.shape
    # step is the batch size, approximately step = 12 requires 10 GB memory for GPU
    method = torch.nn.Softmax(dim=1)
    unet.eval()
    with torch.no_grad():
        if direction=="X":
            tumor_distribution_prediction=np.zeros([x_size,y_size,z_size])
            for i in range(0,x_size,1):
                arr=arr_at_each_location(image,direction,i)
                batch_input=torch.from_numpy(arr).cuda()
                predict=unet(batch_input)
                predict=method(predict)
                predict=predict.cpu().numpy()[0,1,:,:]
                tumor_distribution_prediction[i,:,:]=predict
        elif direction=="Y":
            tumor_distribution_prediction=np.zeros([x_size,y_size,z_size])
            for i in range(0,y_size,1):
                arr=arr_at_each_location(image,direction,i)
                batch_input=torch.from_numpy(arr).cuda()
                predict=unet(batch_input)
                predict=method(predict)
                predict=predict.cpu().numpy()[0,1,:,:]
                tumor_distribution_prediction[:,i,:]=predict
        elif direction=="Z":
            tumor_distribution_prediction=np.zeros([x_size,y_size,z_size])
            for i in range(0,z_size,1):
                arr=arr_at_each_location(image,direction,i)
                batch_input=torch.from_numpy(arr).cuda()
                predict=unet(batch_input)
                predict=method(predict)
                predict=predict.cpu().numpy()[0,1,:,:]
                tumor_distribution_prediction[:,:,i]=predict
        else:
            assert False
    
    return tumor_distribution_prediction


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

def final_prediction(image,best_model_fns,threshold=2,filter_predictions=True,lung_mask=None,direction="all"):
    if direction=="all":
        unet=load_model(best_model_fns['X'])
        x_prediction = get_prediction(unet, 'X', image)
        load_model(best_model_fns['Y'])
        y_prediction = get_prediction(unet, 'Y', image)
        load_model(best_model_fns['Z'])
        z_prediction = get_prediction(unet, 'Z', image)
        pred = np.array(x_prediction+y_prediction+z_prediction > threshold, 'float32')
    elif direction=="X":
        unet=load_model(best_model_fns['X'])
        pred = get_prediction(unet, 'X', image)>threshold
    elif direction=="Y":
        unet=load_model(best_model_fns['Y'])
        pred = get_prediction(unet, 'Y', image)>threshold       
    elif direction=="Z":
        unet=load_model(best_model_fns['Z'])
        pred = get_prediction(unet, 'Z', image)>threshold   
    else:
        assert False
    if filter_predictions:
        pred=scipy.ndimage.maximum_filter(pred,3)
    if type(lung_mask)==np.ndarray:
        pred=pred*lung_mask
    return pred

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