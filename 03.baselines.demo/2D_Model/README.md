# 2D Model
This subfolder contains code for `2D Model`.

## Train
The main program for training is `train.py`. You need to specify `TRAIN_DATA_DIR` and `CHECKPOINT_DIR` in the beginning of the file before running `python train.py`.

## Test
The main program for testing is `test.py`. You need to specify `data_root`, which contains the test data, and `model_path`, which contains the path to the trained model. Then you need to run `test.py` to begin testing.