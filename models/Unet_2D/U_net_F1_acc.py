"""
this code calculate the F1 loss of the segmentation:
F1 score = 2*TP/(FN+FP+2*TP)
"""
import numpy as np


def calculate_f1_score_cpu(prediction, ground_truth, strict=False):
    # this is the patient level acc function.
    # the input for prediction and ground_truth must be the same, and the shape should be [height, width, thickness]
    # the ground_truth should in range 0 and 1

    if np.max(prediction) > 1 or np.min(prediction) < 0:
        print("prediction is not probabilistic distribution")
        exit(0)

    if not strict:
        ground_truth = np.array(ground_truth > 0, 'float32')

        TP = np.sum(prediction * ground_truth)
        FN = np.sum((1 - prediction) * ground_truth)
        FP = np.sum(prediction) - TP
    else:
        difference = prediction - ground_truth

        TP = np.sum(prediction) - np.sum(np.array(difference > 0, 'float32') * difference)
        FN = np.sum(np.array(-difference > 0, 'float32') * (-difference))
        FP = np.sum(np.array(difference > 0, 'float32') * difference)
    eps=1e-6
    F1_score = (2*TP+eps)/(FN+FP+2*TP+eps)
    Precision=(TP+eps)/(TP+FP+eps)
    Recall=(TP+eps)/(TP+FN+eps)
    return Precision, Recall, F1_score


def strict_f1(prediction, ground_truth):
    # this is the patient level acc function.
    # the input for prediction and ground_truth must be the same, and the shape should be [height, width, thickness]
    # the ground_truth should in range 0 and 1

    height = np.shape(ground_truth)[0]
    width = np.shape(ground_truth)[1]
    if not (width == np.shape(prediction)[1] and height == np.shape(prediction)[0]):
        print('shape error')
        exit(0)
    if np.max(prediction) > 1 or np.min(prediction) < 0:
        print("prediction is not probabilistic distribution")
        exit(0)

    difference = prediction - ground_truth

    TP = np.sum(prediction) - np.sum(np.array(difference > 0, 'float32') * difference)
    FN = np.sum(np.array(-difference > 0, 'float32') * (-difference))
    FP = np.sum(np.array(difference > 0, 'float32')*difference)

    F1_score = 2*TP/(FN+FP+2*TP)

    return F1_score, TP, FN, FP
