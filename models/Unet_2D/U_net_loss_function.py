import torch
import torch.nn.functional as F


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

    batch_size,_,width,height=prediction_result.size()
    prediction_result = prediction_result.float()
    ground_truth = ground_truth.float()

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_for_prediction_probability = -softmax_then_log(prediction_result)

    # here class = 0 means normal points, class = 1 means tumor points
    log_for_prediction_probability[:, 0, :, :] = balance_weights[0] * log_for_prediction_probability[:, 0, :, :]*(1-ground_truth[:,0,:,:])
    log_for_prediction_probability[:, 1, :, :] = balance_weights[1] * log_for_prediction_probability[:, 1, :, :]*ground_truth[:,0,:,:]

    return_tensor = log_for_prediction_probability
    return_tensor = torch.sum(return_tensor)

    return return_tensor


def cross_entropy_pixel_wise_2d_binary_with_pixel_weight(pred,gt,weight_map,balance_weights=(1,1)):
    prediction_result=pred.float()
    gt=gt.float()
    weight_map=weight_map.float()
    log_prob=-F.log_softmax(prediction_result,dim=1)
    log_prob[:,0,:,:]=log_prob[:,0,:,:]*(1-gt[:,0,:,:])*weight_map[:,0,:,:]
    log_prob[:,1,:,:]=log_prob[:,1,:,:]*(gt[:,0,:,:])*weight_map[:,1,:,:]
    neg_sum=weight_map[:,0,:,:].sum([1,2])
    assert torch.all(neg_sum>0)
    neg_loss=log_prob[:,0,:,:].sum([1,2])/neg_sum

    pos_sum=weight_map[:,1,:,:].sum([1,2])
    assert torch.all(pos_sum>0)
    pos_loss=log_prob[:,1,:,:].sum([1,2])/pos_sum
    loss=balance_weights[0]*neg_loss.mean()+balance_weights[1]*pos_loss.mean()

    return loss


def cross_entropy_pixel_wise_multi_class(prediction, ground_truth, weight_array, balance_weight=(10, 1)):
    """
    all parameters should on GPU, with float32 data type.
    :param balance_weight: balance_weight for class one, two, three, etc
    :param prediction: [batch_size, class_num, height, width], NOT soft_maxed!
    :param ground_truth: [batch_size, class_num, height, width], each pixel with value [0, 1]
    :param weight_array: [batch_size, class_num, height, width], each pixel with value [0, inf)
    :return: a float with value [0, inf)
    """
    prediction = prediction.float()
    ground_truth = ground_truth.float()

    softmax_then_log = torch.nn.LogSoftmax(dim=1)
    log_prediction_probability = -softmax_then_log(prediction)

    return_tensor = log_prediction_probability * ground_truth * weight_array * 1
    for i in range(len(balance_weight)):
        hyper_weight = balance_weight[i]
        return_tensor[:, i, :, :] = return_tensor[:, i, :, :] * hyper_weight
        loss = torch.sum(return_tensor)

    return loss


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


if __name__=="__main__":
    pred=torch.rand(3,2,512,512)
    gt=torch.rand(3,1,512,512)
    weight_map=torch.randint(0,2,(3,2,512,512))
    print(cross_entropy_pixel_wise_2d_binary_with_pixel_weight(pred,gt,weight_map,balance_weights=(1,1)).item())