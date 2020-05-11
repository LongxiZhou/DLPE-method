# CUDA_VISIBLE_DEVICES='0,1,2,3' python train_2ddense_xgfy.py -batch_size 4  -data /ibex/scratch/projects/c2052/H-DenseUNet/xgfy-data
 python -u train_2ddense_xgfy.py -batch_size 32 -n_threads 32 -n_gpus 8
