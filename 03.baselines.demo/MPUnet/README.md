# MPUNet
This subfolder contains the relavent code for the baseline method `MPUNet`.

## Usage
### Create and activate the conda environment
```
conda create -n test_mp python=3.6
conda activate test_mp
```
### Install dependencies
```
pip install -e 03.baselines.demo/MultiPlanarUNet_26
```
(change tensorflow versions from “2.0.0” to “1.2.0” at the 10-th line and 12-th line in the “requirements.txt” seems won’t cause bug)

### Run
```
cd 03.baselines.demo/MPUnet
python prepare_samples.py
mp predict -f ./test_dataset/images/xgfy-A000042_2020-03-02.npy.nii.gz --out_dir ./test_pred/ --no_argmax --overwrite
```