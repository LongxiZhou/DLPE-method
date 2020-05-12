# MPUNet
This subfolder contains the relavent code for the baseline method `MPUNet`.

## Usage
### Create and activate the conda environment
```
conda create -n test_mp python=3.6
conda activate test_mp
```
### Install dependencies
In the root directory of the repo, run
```
pip install -e 03.baselines.demo/MPUnet/MultiPlanarUNet_26
```
### Data and Checkpoint

Checkpoint on Google Drive: `03.baselines.demo/MPUnet/model/` please respect the folder structure.

Data on Google Drive: `CT_scan_spatial_signal_normalized/` paste these normalized `.npy` arrays into `03.baselines.demo/MPUnet/arrays_raw/`

### Run
```
cd 03.baselines.demo/MPUnet
python prepare_samples.py
mp predict -f ./test_dataset/images/S1_data.npy.nii.gz --out_dir ./test_pred/ --no_argmax --overwrite
```
`--no_argmax` means output real number `S1_data.npy.nii.gz` is a file name of the normalized array in `.nii` format.
