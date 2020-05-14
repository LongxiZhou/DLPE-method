"""Test ImageNet pretrained DenseNet"""
from __future__ import print_function
import sys
# sys.path.insert(0,'Keras-2.0.8')
import multiprocessing as mp
import random
from medpy.io import load
import numpy as np
import argparse
import keras
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
import os
# from keras.utils2.multi_gpu import make_parallel
from skimage.transform import resize
import glob
import scipy
import warnings
K.common.set_image_dim_ordering('tf')


from denseunet import DenseUNet
from loss import weighted_crossentropy_2ddense

#  global parameters
parser = argparse.ArgumentParser(description='Keras 2d denseunet Training')
#  data folder
parser.add_argument('-save_path', type=str, default='Experiments/')
#  other paras
# lizx: batch_size
parser.add_argument('-batch_size', type=int, default=8)
parser.add_argument('-input_size', type=int, default=512)
parser.add_argument('-model_weight', type=str, default='./model/densenet161_weights_tf.h5')
parser.add_argument('-input_cols', type=int, default=3)
parser.add_argument('-n_gpus',type=int,default=1)
#  data augment
parser.add_argument('-mean', type=int, default=48)
parser.add_argument('-n_threads', type=int, default=4)
args = parser.parse_args()
print(args)
MEAN = args.mean



N_TRAINING_SAMPLES=2

def slice_dimension(array,i_min,i_max,dim):
    if i_min>=i_max:
        return np.array([])
    else:
        if i_min>=0:
            pad1=0
            start_idx=i_min
        else:
            pad1=-i_min
            start_idx=0
        if i_max<=array.shape[dim]:
            pad2=0
            end_idx=i_max
        else:
            pad2=i_max-array.shape[dim]
            end_idx=array.shape[dim]
        concat_list=[]
        if pad1>0:
            shape=list(array.shape)
            shape[dim]=pad1
            concat_list.append(np.zeros(shape))
        concat_list.append(np.take(array,range(start_idx,end_idx),dim))
        if pad2>0:
            shape=list(array.shape)
            shape[dim]=pad2
            concat_list.append(np.zeros(shape))
        return np.concatenate(concat_list,dim)
        
def slice_array(array,a_min,a_max,b_min,b_max,c_min,c_max):
    assert len(array.shape)==3
    array=slice_dimension(array,a_min,a_max,0)
    array=slice_dimension(array,b_min,b_max,1)
    array=slice_dimension(array,c_min,c_max,2)
    return array

def load_seq_crop_data_masktumor_try(count):
    global data_dict
    img = data_dict["img_list"][count]
    tumor = data_dict["tumor_list"][count]
    minindex = data_dict["minindex_list"][count]
    maxindex = data_dict["maxindex_list"][count]
    num = np.random.randint(0,6)
    if num < 3 : # removed liver_list
        lines = data_dict["liverlines"][count]
        numid = data_dict["liveridx"][count]
    else:
        lines = data_dict["tumorlines"][count]
        numid = data_dict["tumoridx"][count]

    #  randomly scale
    scale = np.random.uniform(0.8,1.2)
    deps = int(args.input_size * scale)
    rows = int(args.input_size * scale)
    cols = 3

    sed = np.random.randint(1,numid)
    cen = lines[sed-1]
    cen = np.fromstring(cen, dtype=int, sep=' ')

    a = min(max(minindex[0] + deps/2, cen[0]), maxindex[0]- deps/2-1)
    b = min(max(minindex[1] + rows/2, cen[1]), maxindex[1]- rows/2-1)
    c_rand=np.random.randint(minindex[2] + cols/2,maxindex[2]- cols/2)
    c = min(max(minindex[2] + cols/2, c_rand), maxindex[2]- cols/2-1)
    # lizx:
    # lizx:
    a_min=a-deps/2
    b_min=b - rows / 2
    c_min=c - cols / 2
    a_max=a + deps / 2
    b_max=b + rows / 2
    c_max=c + cols/2+1

    # print("range",a_min,a_max,b_min,b_max,c_min,c_max)
    # print("c range",minindex[2] + cols/2, maxindex[2]- cols/2-1)

    cropp_img=slice_array(img,a_min,a_max,b_min,b_max,c_min,c_max)
    cropp_tumor=slice_array(tumor,a_min,a_max,b_min,b_max,c_min,c_max)

    cropp_img -= MEAN
     # randomly flipping
    flip_num = np.random.randint(0, 8)
    if flip_num == 1:
        # problem
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
    elif flip_num == 2:
        # problem
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    elif flip_num == 3:
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 4:
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 5:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
    elif flip_num == 6:
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
        cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
        cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
    elif flip_num == 7:
        cropp_img = np.flipud(cropp_img)
        cropp_tumor = np.flipud(cropp_tumor)
        cropp_img = np.fliplr(cropp_img)
        cropp_tumor = np.fliplr(cropp_tumor)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        cropp_tumor = resize(cropp_tumor, (args.input_size,args.input_size,args.input_cols), order=0, mode='edge', cval=0, clip=True, preserve_range=True)
        cropp_img   = resize(cropp_img, (args.input_size,args.input_size,args.input_cols), order=3, mode='constant', cval=0, clip=True, preserve_range=True)
    return cropp_img, cropp_tumor[:,:,1]
def load_seq_crop_data_masktumor_evaluate(img,index):
    assert 0<=index<img.shape[2]

    #  randomly scale
    deps = args.input_size
    rows = args.input_size
    cols = 3

    # lizx:
    a_min=0
    b_min=0
    c_min=index - cols / 2
    a_max=img.shape[0]
    b_max=img.shape[1]
    c_max=index + cols/2+1

    cropp_img = slice_array(img,a_min,a_max,b_min,b_max,c_min,c_max).copy()
    cropp_img -= MEAN
    return cropp_img

def generate_arrays_from_file(batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list,steps_per_epoch):
    global data_dict
    data_dict={
        'trainidx':trainidx,
        'img_list':img_list,
        'tumor_list':tumor_list,
        'liverlines':liverlines,
        'liveridx':liveridx,
        'tumorlines':tumorlines,
        'tumoridx':tumoridx,
        'minindex_list':minindex_list,
        'maxindex_list':maxindex_list
    }
    if args.n_threads>0:
        pool=mp.Pool()
    try:
        while True:
            X = np.zeros((batch_size, args.input_size, args.input_size, args.input_cols), dtype='float32')
            Y = np.zeros((batch_size, args.input_size, args.input_size, 1), dtype='int16')
            Parameter_List = []
            for idx in xrange(batch_size):
                count = random.choice(trainidx)
                Parameter_List.append(count)
            if args.n_threads>0:
                result_list = pool.map(load_seq_crop_data_masktumor_try, Parameter_List)
            else:
                result_list= map(load_seq_crop_data_masktumor_try, Parameter_List)
            for idx in xrange(len(result_list)):
                X[idx, :, :, :] = result_list[idx][0]
                Y[idx, :, :, 0] = result_list[idx][1]
            yield (X,Y)
    finally:
        if args.n_threads>0:
            pool.close()
            pool.join()


def load_fast_files(args):
    train_dir="xgfy_data/myTrainingData"
    test_dir= "xgfy_data/myTestData"
    train_txt_dir="xgfy_data/myTrainingDataTxt"
    lung_mask_txt_dir="xgfy_data/myTrainingDataTxt/lung_mask"
    label_txt_dir="xgfy_data/myTrainingDataTxt/label"
    box_txt_dir="xgfy_data/myTrainingDataTxt/box"

    img_list = []
    tumor_list = []
    minindex_list = []
    maxindex_list = []
    tumorlines = []
    tumoridx = []
    liveridx = []
    liverlines = []
    volume_paths=glob.glob(os.path.join(train_dir,"*_volume.npy"))
    # TODO
    volume_paths=get_files()
    trainidx = list(range(len(volume_paths)))

    print('-'*30)
    print("loading data")
    print('-'*30)

    for i,volume_path in enumerate(volume_paths):
        print("[%d/%d]\r"%(i+1,len(volume_paths)))
        noext=os.path.splitext(os.path.basename(volume_path))[0]
        fid=noext.replace('_volume','')
        img=np.load(volume_path,mmap_mode='r')
        tumor=np.load(volume_path.replace('_volume','_segmentation-and-lung-mask'),mmap_mode='r')
        img_list.append(img)
        tumor_list.append(tumor)
        maxmin = np.loadtxt(os.path.join(box_txt_dir,fid+".txt"))
        minindex = maxmin[0:3]
        maxindex = maxmin[3:6]
        minindex = np.array(minindex, dtype='int')
        maxindex = np.array(maxindex, dtype='int')
        minindex[0] = max(minindex[0] - 3, 0)
        minindex[1] = max(minindex[1] - 3, 0)
        minindex[2] = max(minindex[2] - 3, 0)
        maxindex[0] = min(img.shape[0], maxindex[0] + 3)
        maxindex[1] = min(img.shape[1], maxindex[1] + 3)
        maxindex[2] = min(img.shape[2], maxindex[2] + 3)
        minindex_list.append(minindex)
        maxindex_list.append(maxindex)
        f1 = open(os.path.join(label_txt_dir,fid+".txt"), 'r')
        tumorline = f1.readlines()
        tumorlines.append(tumorline)
        tumoridx.append(len(tumorline))
        f1.close()
        f2 = open(os.path.join(lung_mask_txt_dir,fid+".txt"), 'r')
        liverline = f2.readlines()
        liverlines.append(liverline)
        liveridx.append(len(liverline))
        f2.close()

    return trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list

def get_prediction(model,img,batch_size=args.batch_size):
    pred_prob=np.zeros_like(img)
    pred_mask=np.zeros_like(img)
    for i in range(0,img.shape[2],batch_size):
        print("[%d/%d]"%(i+1,img.shape[2]))
        inp=np.zeros((batch_size,img.shape[0],img.shape[1],3))
        for j in range(batch_size):
            if i+j<img.shape[2]:
                inp[j,:,:,:]=load_seq_crop_data_masktumor_evaluate(img,i+j)
            else:
                inp[j,:,:,:]=np.zeros((img.shape[0],img.shape[1],3))
        prediction=model.predict(inp,batch_size=batch_size)
        prediction_shifted=prediction-np.max(prediction,axis=3,keepdims=True)
        prediction_prob=np.exp(prediction_shifted[:,:,:,2])/np.exp(prediction_shifted).sum(axis=3)
        prediction_seg=prediction.argmax(axis=3)
        prediction_mask=(prediction_seg==2).astype(np.float32)
        pred_prob[:,:,i:(min(i+batch_size,img.shape[2]))]=prediction_prob[0:(min(batch_size,img.shape[2]-i)),:,:].transpose((1,2,0))
        pred_mask[:,:,i:(min(i+batch_size,img.shape[2]))]=prediction_mask[0:(min(batch_size,img.shape[2]-i)),:,:].transpose((1,2,0))
    return pred_prob,pred_mask

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

def train():

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    if args.n_gpus>1:
        print("Using %d GPUs"%(args.n_gpus))
        with tf.device("/cpu:0"):
            model = DenseUNet(reduction=0.5)
        model.load_weights(args.model_weight, by_name=True)
        model=keras.utils.multi_gpu_model(model, gpus=args.n_gpus, cpu_merge=True, cpu_relocation=False)
    else:
        print("Using single GPU"%(args.n_gpus))
        model = DenseUNet(reduction=0.5)
        model.load_weights(args.model_weight, by_name=True)

    sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=[weighted_crossentropy_2ddense])
    trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list = load_fast_files(args)

    print('-'*30)
    print('Fitting model......')
    print('-'*30)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if not os.path.exists(args.save_path + "/model"):
        os.mkdir(args.save_path + '/model')
        os.mkdir(args.save_path + '/history')
    else:
        if os.path.exists(args.save_path+ "/history/lossbatch.txt"):
            os.remove(args.save_path + '/history/lossbatch.txt')
        if os.path.exists(args.save_path + "/history/lossepoch.txt"):
            os.remove(args.save_path + '/history/lossepoch.txt')

    slices_per_slide=250
    n_slides=len(trainidx)
    steps_per_epoch=int(slices_per_slide*n_slides/args.batch_size)
    print("slices_per_slide:",slices_per_slide,"steps_per_epoch:",steps_per_epoch)

    model_checkpoint = ModelCheckpoint(args.save_path + '/model/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose = 1,
                                       save_best_only=False,save_weights_only=False,mode = 'min', period = 1)

    with warnings.catch_warnings():
        warnings.filterwarnings('default',message='.*',category=UserWarning)
        model.fit_generator(generate_arrays_from_file(args.batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx,
                                                    liveridx, minindex_list, maxindex_list,steps_per_epoch),steps_per_epoch=steps_per_epoch, # TODO
                                                        epochs= 20, verbose = 1, callbacks = [model_checkpoint], max_queue_size=10,
                                                        workers=1, use_multiprocessing=False)
    print ('Finished Training .......')

def calculate_z_mean():
    box_dir="./xgfy_data/myTrainingDataTxt/box"
    boxes=[]
    for box_fn in os.listdir(box_dir):
        box=np.loadtxt(os.path.join(box_dir,box_fn))
        boxes.append(box)
    boxes=np.stack(boxes,0)
    print((boxes[:,5]-boxes[:,2]).mean())
def get_files():
    original_dir="/ibex/scratch/projects/c2052/COVID-19/check_augmentation_effect/raw_augmentation"
    original_fns=os.listdir(original_dir)
    original_slide_ids=[os.path.splitext(fn)[0] for fn in original_fns]

    train_dir="xgfy_data/myTrainingData"
    volume_paths=[os.path.join(train_dir,"%s_volume.npy"%(id)) for id in original_slide_ids]
    volume_paths=[p for p in volume_paths if os.path.isfile(p)]

    aug_dir="xgfy_data/myTrainingData"
    volume_paths+=glob.glob(os.path.join(aug_dir,"*_3_volume.npy"))
    return volume_paths

def test_generator():
    trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx, liveridx, minindex_list, maxindex_list = load_fast_files(args)
    slices_per_slide=250
    n_slides=len(trainidx)
    steps_per_epoch=int(slices_per_slide*n_slides/args.batch_size)
    it=generate_arrays_from_file(args.batch_size, trainidx, img_list, tumor_list, tumorlines, liverlines, tumoridx,
                                                    liveridx, minindex_list, maxindex_list,steps_per_epoch)
    x,y=next(it)

def predict():
    from lib.custom_layers import Scale
    test_arr_dir="/ibex/scratch/projects/c2052/COVID-19/arrays_raw"
    model_file="./Experiments/model/weights.07-0.16.hdf5"
    model=keras.models.load_model(
        model_file,
        custom_objects={'Scale':Scale},
        compile=False\
    )
    prediction_dir="prediction-aug"
    # TODO
    for fn in os.listdir(test_arr_dir):
        print(fn)
        array=np.load(os.path.join(test_arr_dir,fn))
        if len(array.shape)==3:
            image=array
        elif len(array.shape)==4:
            image=array[:,:,:,0]
            gt=array[:,:,:,1]
        pred_prob,pred_mask=get_prediction(model,image,batch_size=args.batch_size)
        lung_mask=np.ones_like(image)
        if len(array.shape)==4:
            f1_score_evaluation(pred_mask,gt, lung_mask, filter_ground_truth=True)
        np.save(os.path.join(prediction_dir,"pred_prob-%s"%(fn)),pred_prob)
        np.save(os.path.join(prediction_dir,"pred_mask-%s"%(fn)),pred_mask)
if __name__ == '__main__':
    predict()