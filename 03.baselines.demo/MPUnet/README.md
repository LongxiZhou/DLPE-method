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
### Run
```
cd 03.baselines.demo/MPUnet
python prepare_samples.py
mp predict -f ./test_dataset/images/xgfy-A000042_2020-03-02.npy.nii.gz --out_dir ./test_pred/ --no_argmax --overwrite
```