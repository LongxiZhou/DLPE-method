## Usage

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

Checkpoint on Google Drive: `03.baselines.demo/3D_Unet/pytorch3dunet/3dunet/` please respect the folder structure.
Data on Google Drive: `CT_scan_spatial_signal_normalized/` paste these normalized `.npy` arrays into `03.baselines.demo/3D_Unet/arrays_raw/`

### Transfer .npy to .h5 files

The normalized arrays are in `.npy` format, while 3D U-net takes `.h5` as input. 

see `./03.baselines.demo/3D U-net/pre_process.py` this file convert normalized arrays into `.h5` format

In `pre_process.py` you need to change the `"main_data_path"` to the directory of `.npy` and the `"destin_path"` to the directory of `.h5`.

```
cd ./03.baselines.demo/3D_Unet
python pre_process.py
```
Now the testing `.npy` files have been changed into the file format for 3D Unet, which is in `.h5`

### Test 3D Unet

please open `./03.baselines.demo/3D_Unet/resources/test_config_ce.yaml`, line 2 is the model_path, and you should work at `./3D_Unet`

please open `./03.baselines.demo/3D_Unet/resources/test_config_ce.yaml`, line 41 is the directory saving proprocessed `.h5` files ready for prediction, you must change it to your own directory, e.g. `/home/zhoul0a/Desktop/COVID-19/models/3D_Unet/h5files/`

```
python pytorch3dunet/predict.py --config ./03.baselines.demo/3D_Unet/resources/test_config_ce.yaml
```

The predictions will be saved to the folder containing the `.h5` test sets, which is the `"destin_path"` you have specified in `Transfer .npy to .h5 files`.


### Post-processing (convert .h5 to .npz for further evaluation)

```
python post_process.py
```

The codes are modified based on [/wolny/pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).
