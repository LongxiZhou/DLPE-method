import os
import models.Unet_2D.train as run_model
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
ibex = False
if not ibex:
    top_directory = '/home/zhoul0a/Desktop/'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'  # use two V100 GPU
else:
    top_directory = '/ibex/scratch/projects/c2052/'

TRAIN_DATA_DIR = "/home/zhoul0a/Desktop/Sep/traning_sample/"
# each sample is a array with path like: TRAIN_DATA_DIR + 'X/X_123_patient-id_time.npy'
WEIGHT_DIR = "/ibex/scratch/projects/c2052/blood_vessel_seg/"
# each sample has a weight array with path like: WEIGHT_DIR + "balance_weight_array/X/weights_X_123_patient-id_time.npy"
CHECKPOINT_DIR = "/home/zhoul0a/Desktop/Sep/check_point/"

parameters = {
    "n_epochs": 3000,  # this is the maximum epoch
    "batch_size": 64,
    "lr": 1e-4,
    "channels": 3,  # the input channel of the sample: window_width * data_channels + enhanced_channels
    'workers': 32,  # num CPU for the parallel data loading
    "balance_weights": None,  # give a bound during training
    "train_data_dir": None,
    "weight_dir": None,
    "test_data_dir": None,  # use five-fold test, so the train data dir is the same with the test data dir.
    "checkpoint_dir": None,
    "saved_model_filename": None,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": 0,
    "wrong_patient_id": [],  # ['xwqg-A00085', 'xwqg-A00121', 'xwqg-B00027', 'xwqg-B00034'],
    "best_f1": None,
    "init_features": 16,
    "beta": 1.5,  # number times recall is more important than precision.
    "target_performance": 0.93,  # what does "high performance" mean for [precision, recall]
    "flip_remaining:": 1
}


def modify_params(direction='X', rim_enhanced=False, test_id=0):
    parameters["test_id"] = test_id
    parameters["saved_model_filename"] = str(test_id) + "_saved_model.pth"
    train_dict = TRAIN_DATA_DIR + direction
    parameters["train_data_dir"] = train_dict
    parameters["test_data_dir"] = train_dict
    if rim_enhanced:
        weight_dict = WEIGHT_DIR + "balance_weight_array_rim_enhance/" + direction
        check_point_dict = CHECKPOINT_DIR + "balance_weight_array_rim_enhance/" + direction
    else:
        weight_dict = WEIGHT_DIR + "balance_weight_array/" + direction
        check_point_dict = CHECKPOINT_DIR + "balance_weight_array/" + direction
    parameters["weight_dir"] = weight_dict
    parameters["checkpoint_dir"] = check_point_dict
    if parameters["balance_weights"] is None:
        parameters["balance_weights"] = [1000000000, 1]


def training_one_direction(direction, rim_enhanced=False):
    for test_id in range(5):
        modify_params(direction, rim_enhanced, test_id)
        print('directing:', direction, "rim_enhanced:", rim_enhanced, 'test_id', test_id)
        run_model.training(parameters)


def training_one_test_id(test_id, rim_enhanced=False):
    for direction in ['X', 'Y', 'Z']:
        modify_params(direction, rim_enhanced, test_id)
        print('directing:', direction, "rim_enhanced:", rim_enhanced, 'test_id', test_id)
        run_model.training(parameters)


def training_all_direction(rim_enhanced=False):
    training_one_direction('X', rim_enhanced)
    training_one_direction('Y', rim_enhanced)
    training_one_direction('Z', rim_enhanced)


def training_all_test_id(rim_enhanced=False):
    for test_id in range(5):
        training_one_test_id(test_id, rim_enhanced)


training_one_test_id(0)
training_one_test_id(1)
training_one_test_id(2)
