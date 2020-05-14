from medpy.io import load, save
import os
import os.path
import numpy as np
import shutil
import glob
import pprint
N_TRAINING_SAMPLES=2
def proprecessing(image_path, save_folder):
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)
    filelist = os.listdir(image_path)
    filelist = [item for item in filelist if 'volume' in item]
    for file in filelist:
        img, img_header = load(image_path+file)
        img[img < -200] = -200
        img[img > 250] = 250
        img = np.array(img, dtype='float32')
        print ("Saving image "+file)
        save(img, "./data/" + save_folder + file)
def preprocessing_xgfy(data_dir,lung_mask_dir,save_dir,mode="train"):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # TODO
    # if mode=="train":
    #     filelist1=[fn for fn in os.listdir(data_dir) if int(fn.split('_')[0][-5:])%5 in [0,1,2,3]]
    # elif mode=="test":
    #     filelist1=[fn for fn in os.listdir(data_dir) if int(fn.split('_')[0][-5:])%5 in [4]]
    filelist1=os.listdir(data_dir)
    pprint.pprint(filelist1)
    filelist2=os.listdir(lung_mask_dir)
    for file in filelist1:
        if file not in filelist2:
            continue
        noext=os.path.splitext(file)[0]
        data1=np.load(os.path.join(data_dir,file))
        data2=np.load(os.path.join(lung_mask_dir,file))
        img=data1[:,:,:,0]
        label=data1[:,:,:,1]
        lung_mask=data2[:,:,:,1]
        print("saving %s"%(noext))
        segmentation_and_lung_mask=np.zeros_like(label)
        segmentation_and_lung_mask[lung_mask>0.5]=1
        segmentation_and_lung_mask[label>0.5]=2
        np.save(os.path.join(save_dir,"%s_segmentation-and-lung-mask"%(noext)),segmentation_and_lung_mask)
        np.save(os.path.join(save_dir,"%s_volume"%(noext)),img)
        np.save(os.path.join(save_dir,"%s_segmentation"%(noext)),label)
        np.save(os.path.join(save_dir,"%s_lung-mask"%(noext)),lung_mask)
def copy_segmentation(seg_path,save_folder):
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)
    filelist = os.listdir(seg_path)
    filelist = [item for item in filelist if 'segmentation' in item]
    for file in filelist:
        print ("Saving segmentation "+file)
        shutil.copyfile(seg_path+file,"./data/"+save_folder+file)

def generate_livertxt(image_path, save_folder):
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)

    # Generate Livertxt
    if not os.path.exists("data/"+save_folder+'LiverPixels'):
        os.mkdir("data/"+save_folder+'LiverPixels')

    for i in range(0,N_TRAINING_SAMPLES):
        livertumor, header = load(image_path+'segmentation-'+str(i)+'.nii')
        f = open('data/' +save_folder+'/LiverPixels/liver_' + str(i) + '.txt', 'w')
        index = np.where(livertumor==1)
        x = index[0]
        y = index[1]
        z = index[2]
        np.savetxt(f, np.c_[x,y,z], fmt="%d")
        f.write("\n")
        f.close()

def generate_lungtxt(lung_mask_dir,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path_list=glob.glob(os.path.join(lung_mask_dir,"*_lung-mask.npy"))
    for f_path in f_path_list:
        print("processing %s"%(f_path))
        lung_mask=np.load(f_path)
        noext=os.path.splitext(os.path.basename(f_path))[0]
        fid=noext.replace('_lung-mask','')
        f = open(os.path.join(save_dir,fid+".txt"),'w')
        index = np.where(lung_mask>0.5)
        x = index[0]
        y = index[1]
        z = index[2]
        np.savetxt(f, np.c_[x,y,z], fmt="%d")
        f.write("\n")
        f.close()

def generate_tumortxt(image_path, save_folder):
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)
    # Generate Livertxt
    if not os.path.exists("data/"+save_folder+'TumorPixels'):
        os.mkdir("data/"+save_folder+'TumorPixels')
    for i in range(0,N_TRAINING_SAMPLES):
        livertumor, header = load(image_path+'segmentation-'+str(i)+'.nii')
        f = open("data/"+save_folder+"/TumorPixels/tumor_"+str(i)+'.txt','w')
        index = np.where(livertumor==2)
        x = index[0]
        y = index[1]
        z = index[2]

        np.savetxt(f,np.c_[x,y,z],fmt="%d")

        f.write("\n")
        f.close()

def generate_labeltxt(label_dir,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path_list=glob.glob(os.path.join(label_dir,"*_segmentation.npy"))
    for f_path in f_path_list:
        print("processing %s"%(f_path))
        label=np.load(f_path)
        noext=os.path.splitext(os.path.basename(f_path))[0]
        fid=noext.replace('_segmentation','')
        f = open(os.path.join(save_dir,fid+".txt"),'w')
        index = np.where(label>0.5)
        x = index[0]
        y = index[1]
        z = index[2]
        np.savetxt(f, np.c_[x,y,z], fmt="%d")
        f.write("\n")
        f.close()

def generate_txt(image_path, save_folder):
    if not os.path.exists("data/"+save_folder):
        os.mkdir("data/"+save_folder)
    # Generate Livertxt
    if not os.path.exists("data/"+save_folder+'LiverBox'):
        os.mkdir("data/"+save_folder+'LiverBox')
    for i in range(0,N_TRAINING_SAMPLES):
        values = np.loadtxt('data/myTrainingDataTxt/LiverPixels/liver_' + str(i) + '.txt', delimiter=' ', usecols=[0, 1, 2])
        a = np.min(values, axis=0)
        b = np.max(values, axis=0)
        box = np.append(a,b, axis=0)

        np.savetxt('data/myTrainingDataTxt/LiverBox/box_'+str(i)+'.txt', box,fmt='%d')

def generate_lung_box(lung_mask_txt_dir,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    f_path_list=glob.glob(os.path.join(lung_mask_txt_dir,"*.txt"))
    for f_path in f_path_list:
        print("processing %s"%(f_path))
        noext=os.path.splitext(os.path.basename(f_path))[0]
        fid=noext.replace('_lung-mask','')
        values=np.loadtxt(f_path,delimiter=' ',usecols=[0,1,2])
        a = np.min(values, axis=0)
        b = np.max(values, axis=0)
        box = np.append(a,b, axis=0)
        np.savetxt(os.path.join(save_dir,fid+'.txt'),box,fmt='%d')

def main1():
    proprecessing(image_path='data/TrainingData/', save_folder='myTrainingData/')
    proprecessing(image_path='data/TestData/', save_folder='myTestData/')
    copy_segmentation(seg_path='data/TrainingData/', save_folder='myTrainingData/')
    copy_segmentation(seg_path='data/TestData/', save_folder='myTestData/')
    print ("Generate liver txt ")
    generate_livertxt(image_path='data/TrainingData/', save_folder='myTrainingDataTxt/')
    print ("Generate tumor txt")
    generate_tumortxt(image_path='data/TrainingData/', save_folder='myTrainingDataTxt/')
    print ("Generate liver box ")
    generate_txt(image_path='data/TrainingData/', save_folder='myTrainingDataTxt/')

def main2():
    data_dir="aug-tmp/arrays"
    lung_mask_dir="aug-tmp/lung_masks"
    train_dir="xgfy_data_aug/myTrainingData"
    test_dir= "xgfy_data_aug/myTestData"
    train_txt_dir="xgfy_data_aug/myTrainingDataTxt"
    lung_mask_txt_dir="xgfy_data_aug/myTrainingDataTxt/lung_mask"
    label_txt_dir="xgfy_data_aug/myTrainingDataTxt/label"
    box_txt_dir="xgfy_data_aug/myTrainingDataTxt/box"
    if not os.path.isdir("xgfy_data_aug"):
        os.mkdir("xgfy_data_aug")
    preprocessing_xgfy(
        data_dir=data_dir,
        lung_mask_dir=lung_mask_dir,
        save_dir=train_dir,
        mode="train"
    )
    # preprocessing_xgfy(
    #     data_dir=data_dir,
    #     lung_mask_dir=lung_mask_dir,
    #     save_dir=test_dir,
    #     mode="test"
    # )
    if not os.path.isdir(train_txt_dir):
        os.mkdir(train_txt_dir)

    generate_lungtxt(train_dir,lung_mask_txt_dir)

    generate_lung_box(lung_mask_txt_dir,box_txt_dir)

    generate_labeltxt(train_dir,label_txt_dir)
if __name__=="__main__":
    main2()
