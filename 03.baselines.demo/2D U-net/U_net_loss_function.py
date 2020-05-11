import torch


def cross_entropy_pixel_wise_2d(prediction_result, ground_truth, balance_weights=(1.0, 524.46)):  # 241.46 for hayida
    # prediction_result is a tensor with shape [batch_size, class_num, height, width]
    # prediction_result is not being softmaxed. e.g. prediction_result[1, :, 34, 45] = [-2, 34, 42, 983, -98] (5classes)
    # ground_truth is a tensor with shape [batch_size, class_num, height, width] (same with prediction_result)
    # the ground truth is stored in one-hot format e.g. ground_truth[1, :, 34, 45] = [0, 0, 0, 1, 0]
    # balance_weights. e.g. 5% pixels in train is class 1, so class 1 has weight 1/0.05 = 20
    # balance_weights is iterable with length = class_num, like [class_1_weight, class_2_weight, ...]
    # cross_entropy is calculated pixels-wise, then sum together according to the balance_weights
    # return: the sum together cross_entropy

    prediction_result = prediction_result.float()
    ground_truth = ground_truth.float()

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_for_prediction_probability = -softmax_then_log(prediction_result)

    class_num = prediction_result.shape[1]

    # here class = 0 means normal points, class = 1 means tumor points
    for i in range(class_num):
        log_for_prediction_probability[:, i, :, :] = balance_weights[i] * log_for_prediction_probability[:, i, :, :]

    return_tensor = log_for_prediction_probability * ground_truth
    return_tensor = torch.sum(return_tensor)

    return return_tensor

def cross_entropy_pixel_wise_2d_binary(prediction_result, ground_truth, balance_weights=(1.0, 524.46)):  # 241.46 for hayida
    # prediction_result is a tensor with shape [batch_size, class_num, height, width]
    # prediction_result is not being softmaxed. e.g. prediction_result[1, :, 34, 45] = [-2, 34, 42, 983, -98] (5classes)
    # ground_truth is a tensor with shape [batch_size, class_num, height, width] (same with prediction_result)
    # the ground truth is stored in one-hot format e.g. ground_truth[1, :, 34, 45] = [0, 0, 0, 1, 0]
    # balance_weights. e.g. 5% pixels in train is class 1, so class 1 has weight 1/0.05 = 20
    # balance_weights is iterable with length = class_num, like [class_1_weight, class_2_weight, ...]
    # cross_entropy is calculated pixels-wise, then sum together according to the balance_weights
    # return: the sum together cross_entropy

    prediction_result = prediction_result.float()
    ground_truth = ground_truth.float()

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_for_prediction_probability = -softmax_then_log(prediction_result)

    class_num = prediction_result.shape[1]

    # here class = 0 means normal points, class = 1 means tumor points
    log_for_prediction_probability[:, 0, :, :] = balance_weights[0] * log_for_prediction_probability[:, 0, :, :]*(1-ground_truth[:,0,:,:])
    log_for_prediction_probability[:, 1, :, :] = balance_weights[1] * log_for_prediction_probability[:, 1, :, :]*ground_truth[:,0,:,:]

    return_tensor = log_for_prediction_probability
    return_tensor = torch.sum(return_tensor)

    return return_tensor

def cross_entropy_pixel_wise_3d(prediction_result, ground_truth, balance_weights=(1.0, 2500)):
    # prediction_result is a tensor with shape [batch_size, class_num, height, width, thickness]
    # prediction_result is the probability map of tumors. all values must in range [0, 1]
    # ground_truth is a tensor with shape [batch_size, class_num, height, width, thickness]
    # the ground truth is stored in one-hot format e.g. ground_truth[1, :, 34, 45] = [0, 0, 0, 1, 0]
    # balance_weights. e.g. 5% pixels in train is class 1, so class 1 has weight 1/0.05 = 20
    # balance_weights is iterable with length = class_num, like [class_1_weight, class_2_weight, ...]
    # cross_entropy is calculated pixels-wise, then sum together according to the balance_weights
    # return: the sum together cross_entropy

    prediction_result = prediction_result.float()
    ground_truth = ground_truth.float()

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_for_prediction_probability = -softmax_then_log(prediction_result)

    class_num = prediction_result.shape[1]

    # here class = 0 means normal points, class = 1 means tumor points
    for i in range(class_num):
        log_for_prediction_probability[:, i, :, :, :] = balance_weights[i] * log_for_prediction_probability[:, i, :, :, :]

    return_tensor = log_for_prediction_probability * ground_truth
    return_tensor = torch.sum(return_tensor)

    return return_tensor

