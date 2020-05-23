## Usage

### Install Dependencies

```
conda create -n 3d_unet -c conda-forge -c awolny python=3.7 pytorch-3dunet
source activate 3d_unet

```

Some new folders need to be created.

```
mkdir ./03.baselines.demo/3D\ U-net/arrays_raw/
mkdir ./03.baselines.demo/3D\ U-net/pytorch3dunet/data/
mkdir ./03.baselines.demo/3D\ U-net/pytorch3dunet/predict_h5/
mkdir ./03.baselines.demo/3D\ U-net/pytorch3dunet/predict_npz/
```

### Data and Checkpoint

Checkpoint on Google Drive: `03.baselines.demo/3D_Unet/pytorch3dunet/3dunet/` paste this checkpoint file `3d_unet_checkpoint.pytorch` into `03.baselines.demo/3D U-net/pytorch3dunet/`

Data on Google Drive: `CT_scan_spatial_signal_normalized/` paste these normalized `.npy` arrays into `03.baselines.demo/3D U-net/arrays_raw/`

### Transfer .npy to .h5 files



The normalized arrays are in `.npy` format, while 3D U-net takes `.h5` as input. 

see `./03.baselines.demo/3D U-net/pre_process.py` this file convert normalized arrays into `.h5` format

The output h5 files will be put into `./03.baselines.demo/3D\ U-net/pytorch3dunet/data/`

```
cd ./03.baselines.demo/3D\ U-net/
python pre_process.py
```
Now the testing `.npy` files have been changed into the file format for 3D Unet, which is in `.h5`

### Test 3D Unet


```
cd ./pytorch3dunet/
python predict.py --config ../resources/test_config_ce.yaml
```

The final prediction files will be put into `./predict_npz/`. Each file is a probability map.



The codes are modified based on [/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).
