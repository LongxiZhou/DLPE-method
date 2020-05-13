## Usage


###Description

This 3D V-Net is pretty similar to 3D U-Net. Therefore, the codes of 3D V-Net are revised from the 3D U-Net codes.
The main difference these two ararchitectures contain: 1. Loss function (3D V-Net uses dice loss). 2. 3D V-Net uses the residual block.
These difference are shown in the configuration file (.yaml file).

### Install Dependencies

```
conda create --name 3d_unet --file environment.txt
source activate 3d_unet
pip install nibabel
pip install opencv-python
pip install pydicom
pip install SimpleITK
```

### Data and Checkpoint

Checkpoint on Google Drive: `03.baselines.demo/3D_Vnet/pytorch3dunet/3dunet/` please respect the folder structure.
Data on Google Drive: `CT_scan_spatial_signal_normalized/` paste these normalized `.npy` arrays into `03.baselines.demo/3D_Vnet/arrays_raw/`

### Pre-processing (transfer npy to h5)

```
cd ./03.baselines.demo/3D_Vnet
python preprocessing.py
```
Now the testing `.npy` files have been changed into the file format for 3D V-Net, which is in `.h5`

### Test 3D V-Net

please open `./03.baselines.demo/3D_Vnet/resources/test_config_dice.yaml`, line 2 is the model_path, and you should work at `./3D_Vnet`
please open `./03.baselines.demo//3D_Vnet/resources/test_config_dice.yaml`, line 65 is the directory saving proprocessed `.h5` files ready for prediction, you must change it to your own directory, e.g. `/home/zhoul0a/Desktop/COVID-19/models/3D_Vnet/h5files/`

```
python pytorch3dunet/predict.py --config ./03.baselines.demo/3D_Vnet/resources/test_config_dice.yaml
```

The prediction will be saved to the folder containing test sets.


### Post-processing (transfer h5 to npz for further evaluation)

```
python post_process.py
```



The codes are modified based on [/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).