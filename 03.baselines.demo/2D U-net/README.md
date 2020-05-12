# 2D U-net
This subfolder contains code for `2D U-net`.

## Prediction
1. The environment is exactly the same with that of “02.our.model”
2. put the checkpoint into `./03.baselines.demo/2D U-Net/checkpoint_dirs/checkpoint_dir/`
3. put the normalized arrays into `./03.baselines.demo/2D U-Net/`
The normalized arrays can be got from `./02.our.model/standard/patient_id/time_1/` or
download from google drive. You need to place them in the directory `03.baselines.demo/2D U-net/datasets/arrays_dir`.
4. run `test.py`

## Warning
You need to remove the false positives outside the lungs to get the results in Fig. 4. The lung
masks are stored in `./02.our.model/standard/patient_id/time_1/` or download from google
drive.