# H-DenseUNet

### Introduction

This subfolder contains a modified version of [this repo](https://github.com/xmengli999/H-DenseUNet). The original code has multiple issues when handling multiprocessing and multi-GPU training. We trained the model using 8 V100 GPUs.

### Environment:
	This code is only tested under python2. Check code environment "requirements.txt"

### Usage
1. Data preprocessing:  
   Run:
   ```shell 
   python preprocessing.py 
   ```

2. Train 2D DenseUnet:
    First, you need to download the pretrained model from [ImageNet Pretrained](https://drive.google.com/file/d/1HHiPBKPw539LR0Oj5g1gD3FNRkCsxeGi/view?usp=sharing), extract it and put it in the folder 'model'.
    Then run:
   ```shell
   bash run.sh
   ```

3. Train H-DenseUnet:
    Load your trained model and run   
    
   ```shell
   CUDA_VISIBLE_DEVICES='0' python train_hybrid.py -arch 3dpart
   ```

4. Train H-DenseUnet in end-to-end way:
    
   ```shell
   CUDA_VISIBLE_DEVICES='0' python train_hybrid.py -arch end2end
   ```


