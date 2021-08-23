import numpy as np
import torch
import sys
sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')
import Tool_Functions.Functions as Functions
import models.Unet_2D.U_net_Model as Model
import sample_manager.sample_slicer_multi_classes as slicer


def load_model(check_point_path, array_info, feature_type=None, gpu=True):
    input_channels = array_info['data_channel'] * len(array_info['window']) + array_info['enhanced_channel']
    out_channels = array_info["output_channels"]
    if feature_type is None:
        model = Model.UNet(in_channels=input_channels, out_channels=out_channels, init_features=array_info['init_features'])
    elif feature_type is 'last_cnn':
        print("loading UnetCamLast")
        model = Model.UNetCamLast(in_channels=input_channels, out_channels=out_channels, init_features=array_info['init_features'])
    else:
        print("loading UnetCamBottle")
        model = Model.UNetCamBottle(in_channels=input_channels, out_channels=out_channels,
                                  init_features=array_info['init_features'])
    model_dict = torch.load(check_point_path)["state_dict"]
    if not array_info["mute_output"]:
        print("loading checkpoint")
    model.load_state_dict(model_dict)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if gpu is False:
        device = "cpu"
    if torch.cuda.device_count() > 1 and gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    if not array_info["mute_output"]:
        print("checkpoint loaded with", torch.cuda.device_count(), 'GPU')
    return model


def predict_one_scan_binary_class(rescaled_array, direction, checkpoint_path, array_info, batch_size=16, evaluate=False):
    """
    :param array_info: stores information of the rescaled_array
    :param rescaled_array: float32 array in shape [512, 512, 512, data_channel + enhanced_channel + semantic_channel]
    or in shape [512, 512, 512, data_channel + enhanced_channel]. Like CT data_channel is 1, DCE-MRI data_channel is 6
    :param direction: 'X', 'Y', or 'Z'
    :param checkpoint_path: the check point of the model
    :param batch_size: generally, 1 GB can support 2 samples being predicting simultaneously.
    :param evaluate: whether evaluate the performance?
    :return: the predicted array, float32 in shape [512, 512, 512, semantic_channel]
    """
    model = load_model(checkpoint_path, array_info)

    input_channels = array_info['data_channel'] * len(array_info['window']) + array_info['enhanced_channel']
    samples = slicer.slice_one_direction(rescaled_array, array_info['resolution'], direction,
                                         array_info['data_channel'], array_info['enhanced_channel'], array_info['window'], positive_semantic_channel=array_info['positive_semantic_channel'])
    sample_list = []
    for sample in samples:
        sample_list.append(np.transpose(sample, (2, 0, 1)))
    sample_shape = np.shape(sample_list[0])
    if evaluate:
        print("sample shape is:", sample_shape, "input_channels is:", input_channels, "evaluate:", evaluate)
    sample_array = np.stack(sample_list, axis=0)
    num_samples = len(sample_list)
    prediction_list = []
    index = 0
    soft_max = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        while index < num_samples:
            index_end = index + batch_size
            if index_end >= num_samples:
                index_end = num_samples
            inputs = torch.from_numpy(sample_array[index: index_end, 0: input_channels, :, :]).cuda()
            prediction = model(inputs)
            prediction = soft_max(prediction)
            prediction = prediction.cpu().numpy()
            prediction = np.transpose(prediction, (0, 2, 3, 1))
            prediction_list.append(prediction)
            index = index_end
    prediction_array = np.concatenate(prediction_list, axis=0)
    if direction == 'Y':
        prediction_array = np.swapaxes(prediction_array, 0, 1)
    if direction == 'Z':
        prediction_array = np.swapaxes(prediction_array, 0, 1)
        prediction_array = np.swapaxes(prediction_array, 1, 2)
    if not array_info["mute_output"]:
        print(np.shape(prediction_array))

    if evaluate:
        recall, precision, dice = Functions.evaluate(prediction_array[:, :, :, 1::], rescaled_array[:, :, :, array_info['data_channel'] + array_info['enhanced_channel']::])
        print("recall, precision, dice:", recall, precision, dice)
    return prediction_array[:, :, :, 1]


def predict_one_scan_multi_class(rescaled_array, direction, checkpoint_path, array_info, batch_size=16, evaluate=False):
    """
    :param array_info: stores information of the rescaled_array
    :param rescaled_array: float32 array in shape [512, 512, 512, data_channel + enhanced_channel + semantic_channel]
    or in shape [512, 512, 512, data_channel + enhanced_channel]. Like CT data_channel is 1, DCE-MRI data_channel is 6
    :param direction: 'X', 'Y', or 'Z'
    :param checkpoint_path: the check point of the model
    :param batch_size: generally, 1 GB can support 2 samples being predicting simultaneously.
    :param evaluate: whether evaluate the performance?
    :return: the predicted array, float32 in shape [512, 512, 512, semantic_channel]
    """
    model = load_model(checkpoint_path, array_info)

    input_channels = array_info['data_channel'] * len(array_info['window']) + array_info['enhanced_channel']
    samples = slicer.slice_one_direction(rescaled_array, array_info['resolution'], direction,
                                         array_info['data_channel'], array_info['enhanced_channel'], array_info['window'], positive_semantic_channel=array_info['positive_semantic_channel'])
    sample_list = []
    for sample in samples:
        sample_list.append(np.transpose(sample, (2, 0, 1)))
    sample_shape = np.shape(sample_list[0])
    if evaluate:
        print("sample shape is:", sample_shape, "input_channels is:", input_channels, "evaluate:", evaluate)
    sample_array = np.stack(sample_list, axis=0)
    num_samples = len(sample_list)
    prediction_list = []
    index = 0
    soft_max = torch.nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        while index < num_samples:
            index_end = index + batch_size
            if index_end >= num_samples:
                index_end = num_samples
            inputs = torch.from_numpy(sample_array[index: index_end, 0: input_channels, :, :]).cuda()
            prediction = model(inputs)
            prediction = soft_max(prediction)
            prediction = prediction.cpu().numpy()
            prediction = np.transpose(prediction, (0, 2, 3, 1))
            prediction_list.append(prediction)
            index = index_end
    prediction_array = np.concatenate(prediction_list, axis=0)
    if direction == 'Y':
        prediction_array = np.swapaxes(prediction_array, 0, 1)
    if direction == 'Z':
        prediction_array = np.swapaxes(prediction_array, 0, 1)
        prediction_array = np.swapaxes(prediction_array, 1, 2)
    if not array_info["mute_output"]:
        print(np.shape(prediction_array))

    if evaluate:
        recall, precision, dice = Functions.evaluate(prediction_array[:, :, :, 1::], rescaled_array[:, :, :, array_info['data_channel'] + array_info['enhanced_channel']::])
        print("recall, precision, dice:", recall, precision, dice)
    return prediction_array


def predict_one_sample(sample, model):
    """
    :param: sample should in shape [x, y, input_channels]
    :return: the predicted array, float32 in shape [x, y, semantic_channel]
    """
    assert len(np.shape(sample)) == 3
    input_channels = np.shape(sample)[2]

    sample_list = [np.transpose(sample, (2, 0, 1))]
    sample_array = np.stack(sample_list, axis=0)

    model.eval()
    with torch.no_grad():
        soft_max = torch.nn.Softmax(dim=1)
        inputs = torch.tensor(sample_array[0: 1, 0: input_channels, :, :], requires_grad=False).float().cuda()
        prediction = model(inputs)
        prediction = soft_max(prediction)

    prediction = prediction.cpu().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))

    return prediction[0, :, :, :]


def heat_map_segment(sample, gt, model, gt_channel=1):
    """
    :param: sample should in shape [x, y, input_channels]
    :param: gt should in shape [x, y]
    :return: heat_map: in shape [x, y, input_channels]
    """
    assert len(np.shape(sample)) == 3
    input_channels = np.shape(sample)[2]

    sample_list = [np.transpose(sample, (2, 0, 1))]
    sample_array = np.stack(sample_list, axis=0)

    model.eval()

    soft_max = torch.nn.Softmax(dim=1)
    inputs = torch.tensor(sample_array[0: 1, 0: input_channels, :, :], requires_grad=True, device='cuda').float()
    # inputs = torch.tensor(sample_array[0: 1, 0: input_channels, :, :], requires_grad=True).float().to("cuda:0")
    prediction = model(inputs)

    prediction = soft_max(prediction)
    gt = torch.tensor(gt, requires_grad=False, device='cuda').float()
    total_prob_in_gt = torch.sum(prediction[0, gt_channel, :, :] * gt)

    total_prob_in_gt.backward()

    prediction = prediction.cpu().detach().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))

    heat_map = inputs.grad.cpu().numpy()[0, :, :, :]

    return prediction[0, :, :, gt_channel], np.transpose(heat_map, (1, 2, 0))


def heat_map_cam(sample, gt, model, target_channel=1, version=4):
    """
    version == 1:
    total_prob_in_target = torch.sum(prediction[0, target_channel, :, :] * gt)

    version == 2:
    total_prob_in_target = torch.sum(prediction[0, target_channel, :, :])

    version == 3:
    total_prob_in_target = (torch.sum(prediction_before_softmax[0, target_channel, :, :] * prob_target_map -
    prediction_before_softmax[0, non_target_channels, :, :] * prob_target_map))

    :param: sample should in shape [x, y, input_channels]
    :param: gt should in shape [x, y]
    :return: heat_map: in shape [x, y, input_channels]
    """
    assert len(np.shape(sample)) == 3
    input_channels = np.shape(sample)[2]

    sample_list = [np.transpose(sample, (2, 0, 1))]
    sample_array = np.stack(sample_list, axis=0)

    model.eval()

    soft_max = torch.nn.Softmax(dim=1)
    inputs = torch.tensor(sample_array[0: 1, 0: input_channels, :, :], requires_grad=True, device='cpu').float()

    prediction, feature_layer = model(inputs)

    feature_layer.retain_grad()

    if version == 1:
        prediction = soft_max(prediction)
        gt = torch.tensor(gt, requires_grad=False, device='cpu').float()
        total_prob_in_target = torch.sum(prediction[0, target_channel, :, :] * gt)
    elif version == 2:
        prediction = soft_max(prediction)
        total_prob_in_target = torch.sum(prediction[0, target_channel, :, :])
    elif version == 3:
        prob_map = soft_max(prediction)
        summation_target = torch.sum(2 * prediction[0, target_channel, :, :] * prob_map[0, target_channel, :, :])
        summation_all = torch.sum(prediction[0, :, :, :], dim=1, keepdim=True) * prob_map[0, target_channel, :, :]
        total_prob_in_target = summation_target - torch.sum(summation_all)
    else:
        assert target_channel == 1
        p_t = prediction[0, target_channel, :, :]
        p_n = prediction[0, 0, :, :]
        total_prob_in_target = torch.sum(torch.log(1 + torch.exp(p_t - p_n)))

    total_prob_in_target.backward()

    feature_layer_grad = feature_layer.grad.numpy()[0, :, :, :]

    shape = np.shape(feature_layer_grad)
    channels = shape[0]

    weight_list = []

    for i in range(channels):
        weight_list.append(np.sum(feature_layer_grad[i, :, :]))

    heat_map = np.zeros([shape[1], shape[2]], 'float32')

    for i in range(channels):
        heat_map = heat_map + weight_list[i] * feature_layer_grad[i, :, :]

    heat_map = np.clip(heat_map, 0, np.inf)

    prediction = prediction.cpu().detach().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    return prediction[0, :, :, target_channel], heat_map


def heat_map_bottle(sample, gt, model, gt_channel=1, version=2):
    """
    version == 1:
    total_prob_in_gt = torch.sum(prediction[0, gt_channel, :, :] * gt)

    version == 2:
    total_prob_in_gt = torch.sum(prediction[0, gt_channel, :, :])

    :param: sample should in shape [x, y, input_channels]
    :param: gt should in shape [x, y]
    :return: heat_map: in shape [x, y, input_channels]
    """
    assert len(np.shape(sample)) == 3
    input_channels = np.shape(sample)[2]

    sample_list = [np.transpose(sample, (2, 0, 1))]
    sample_array = np.stack(sample_list, axis=0)

    model.eval()

    soft_max = torch.nn.Softmax(dim=1)
    inputs = torch.tensor(sample_array[0: 1, 0: input_channels, :, :], requires_grad=True, device='cpu').float()
    # inputs = torch.tensor(sample_array[0: 1, 0: input_channels, :, :], requires_grad=True).float().to("cuda:0")
    prediction, last_layer = model(inputs)

    last_layer.retain_grad()

    prediction = soft_max(prediction)
    if version == 1:
        gt = torch.tensor(gt, requires_grad=False, device='cpu').float()
        total_prob_in_gt = torch.sum(prediction[0, gt_channel, :, :] * gt)
    else:
        total_prob_in_gt = torch.sum(prediction[0, gt_channel, :, :])

    total_prob_in_gt.backward()

    last_layer_grad = last_layer.grad.numpy()[0, :, :, :]

    shape = np.shape(last_layer_grad)
    channels = shape[0]

    weight_list = []

    for i in range(channels):
        weight_list.append(np.sum(last_layer_grad[i, :, :]))

    heat_map = np.zeros([shape[1], shape[2]], 'float32')

    for i in range(channels):
        heat_map = heat_map + weight_list[i] * last_layer_grad[i, :, :]

    heat_map = np.clip(heat_map, 0, np.inf)

    prediction = prediction.cpu().detach().numpy()
    prediction = np.transpose(prediction, (0, 2, 3, 1))
    return prediction[0, :, :, gt_channel], heat_map
