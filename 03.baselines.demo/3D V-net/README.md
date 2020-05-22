# 3D V-net
## Install Environment
Create conda environment
```bash
conda create -n 3dvnet
```
Install dependencies
```bash
conda install --file requirements.txt
```
## Run
### Training
```bash
python train.py --train_data_dir <train_dir> --test_data_dir <test_dir> --checkpoint_dir <checkpoint_dir>
```
`<train_dir>` is the directory containing the training files, `<test_dir>` is the directory containing the test files, and `<checkpoint_dir>` is the directory that stores the trained model.

A trained model is stored on the same Google Drive folder provided before.

### Testing
Modify `model_path`, `arrays_dir`, `lung_mask_dir`, `prediction_dir` at the beginning of test.py.
Run:
```bash
python test.py
```
