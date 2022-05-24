import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.transforms
import math
import collections
import sys

sys.path.append('/ibex/scratch/projects/c2052/Lung_CAD_NMI/source_codes')

import models.Unet_2D.U_net_Model as unm
import models.Unet_2D.U_net_loss_function as unlf
from models.Unet_2D.dataset import RandomFlipWithWeight, RandomRotateWithWeight, ToTensorWithWeight, \
    WeightedTissueDataset

parameters = {
    "n_epochs": None,
    "batch_size": None,
    "lr": 1e-4,
    "channels": None,  # the input channel of the sample
    'workers': None,  # num CPU for the parallel data loading
    "balance_weights": None,
    "train_data_dir": None,
    "weight_dir": None,
    "test_data_dir": None,  # use train for test
    "checkpoint_dir": None,
    "saved_model_filename": None,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "test_id": None,
    "wrong_patient_id": None,
    "best_f1": None,
    "init_features": None
}


def save_checkpoint(epoch, model, optimizer, history, best_f1, params=parameters, best=True):
    filename = params["saved_model_filename"]
    if not best:  # this means we store the current model
        filename = "current_" + params["saved_model_filename"]
    filename = os.path.join(params["checkpoint_dir"], filename)
    torch.save({
        'epoch': epoch,
        'state_dict': model.module.state_dict() if type(model) == nn.DataParallel else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history': history,
        'best_f1': best_f1
    }, filename)


def train_loop(model, optimizer, train_loader, test_loader, params=parameters, resume=True):
    if resume and params["best_f1"] is not None:
        best_f1 = params["best_f1"]
    else:
        best_f1 = 0
    saved_model_path = os.path.join(params["checkpoint_dir"], params["saved_model_filename"])
    if resume and os.path.isfile(saved_model_path):
        data_dict = torch.load(saved_model_path)
        epoch_start = data_dict["epoch"]
        if type(model) == nn.DataParallel:
            model.module.load_state_dict(data_dict["state_dict"])
        else:
            model.load_state_dict(data_dict["state_dict"])
        optimizer.load_state_dict(data_dict["optimizer"])
        history = data_dict["history"]
        best_f1 = data_dict["best_f1"]
        print("best_f1 is:", best_f1)
    else:
        epoch_start = 0
        history = collections.defaultdict(list)
    print("Going to train epochs [%d-%d]" % (epoch_start + 1, epoch_start + params["n_epochs"]))

    base = 1

    precision_phase = True  # if in precision phase, the model increase the precision

    flip_remaining = params["flip_remaining:"]  # initially, model has high recall low precision. Thus, we initially in
    # the precision phase. When precision is high and recall is low, we flip to recall phase.
    # flip_remaining is the number times flip precision phase to recall phase

    fluctuate_phase = False
    # when flip_remaining is 0 and the model reached target_performance during recall phase, change to fluctuate phase
    # during this phase, the recall fluctuate around the target_performance.

    fluctuate_epoch = 0

    for epoch in range(epoch_start + 1, epoch_start + 1 + params["n_epochs"]):
        print("Training epoch %d" % (epoch))
        print("fluctuate_epoch:", fluctuate_epoch)
        if fluctuate_epoch > 20:
            break
        model.train()
        for i, sample in enumerate(train_loader):
            current_batch_size, input_channel, width, height = sample["image"].shape
            image = sample["image"].to(params["device"]).float()
            label = torch.zeros([current_batch_size, 2, width, height])
            label[:, 0:1, :, :] = 1 - sample["label"]
            label[:, 1::, :, :] = sample["label"]
            label = label.to(params["device"]).float()
            weight = sample["weight"].to(params["device"]).float()
            pred = model(image)
            maximum_balance = params["balance_weights"][0]
            hyper_balance = base  # math.pow(base, 1.2)
            if hyper_balance > maximum_balance:
                hyper_balance = maximum_balance
            loss = unlf.cross_entropy_pixel_wise_multi_class(pred, label, weight, (hyper_balance, 100))
            # loss = unlf.cross_entropy_pixel_wise_2d_binary_with_pixel_weight(pred, sample["label"].to(parameters["device"]), weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("\tstep [%d/%d], loss=%.4f" % (i + 1, len(train_loader), loss), base, (hyper_balance, 100))
        print("\tEvaluating")
        eval_vals_train = evaluate(model, test_loader, params)

        print("flip_remaining:", flip_remaining, "precision_phase:", precision_phase)

        if epoch > 5 and not fluctuate_phase:  # recall will be very very high, start with precision phase

            if eval_vals_train["precision"] < params["target_performance"] and precision_phase:
                print("precision phase, increase base to", int(base * 1.4 + 1))
                base = int(base * 1.4 + 1)
            elif precision_phase:
                precision_phase = False
                print("change to recall phase")

            if eval_vals_train["recall"] < params["target_performance"] and not precision_phase:
                print("recall phase, decrease base to", int(base / 1.5 - 1))
                base = int(base / 1.5 - 1)
                if base < 1:
                    base = 1
            elif not precision_phase:
                if flip_remaining > 0:
                    flip_remaining -= 1
                    precision_phase = True
                    print("change to precision phase")
                else:
                    print("change to fluctuate phase")
                    fluctuate_phase = True

        if fluctuate_phase:
            if eval_vals_train["recall"] < params["target_performance"]:
                print("fluctuate phase, decrease base to", base / 1.1 - 1)
                base = base / 1.1 - 1
                if base < 1:
                    base = 1
            else:
                print("fluctuate phase, increase base to", base * 1.1 + 1)
                base = base * 1.1 + 1
            fluctuate_epoch += 1

        for k, v in eval_vals_train.items():
            history[k + "_train"].append(v)
        print("\tloss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f"
              % (
                  eval_vals_train["loss"], eval_vals_train["precision"], eval_vals_train["recall"],
                  eval_vals_train["f1"]))
        if eval_vals_train["f1"] > best_f1 and eval_vals_train["recall"] > params["target_performance"] - 0.01:
            print("saving model as:", str(params["test_id"]) + "_saved_model.pth")
            best_f1 = eval_vals_train["f1"]
            save_checkpoint(epoch, model, optimizer, history, best_f1, params)
        save_checkpoint(epoch, model, optimizer, history, eval_vals_train["f1"], params, False)
    print("Training finished")
    print("best_f1:", best_f1)


def evaluate(model, test_loader, params):
    model.eval()
    with torch.no_grad():
        vals = {
            'loss': 0,
            'tp': 0,
            'fp': 0,
            'fn': 0,
            "tot_pixels": 0,
        }
        for i, sample in enumerate(test_loader):
            current_batch_size, input_channel, width, height = sample["image"].shape
            vals["tot_pixels"] += current_batch_size * width * height
            image = sample["image"].to(params["device"]).float()
            label = sample["label"].to(params["device"]).float()
            pred = model(image)
            loss = unlf.cross_entropy_pixel_wise_2d_binary(pred, label,
                                                           balance_weights=params["balance_weights"]).item()
            # here we use not_voxel_weighted loss to illustrate the performance
            vals["loss"] += loss

            pred = (pred[:, 1, :, :] > pred[:, 0, :, :]).float().unsqueeze(1)
            pred_tp = pred * label
            tp = pred_tp.sum().float().item()
            vals["tp"] += tp
            vals["fp"] += pred.sum().float().item() - tp
            pred_fn = (1 - pred) * label
            vals["fn"] += pred_fn.sum().float().item()
        eps = 1e-6
        vals["loss"] = vals["loss"] / vals["tot_pixels"]

        beta = params['beta']  # number times recall is more important than precision

        vals["precision"] = (vals["tp"] + eps) / (vals["tp"] + vals["fp"] + eps)
        vals["recall"] = (vals["tp"] + eps) / (vals["tp"] + vals["fn"] + eps)
        vals["f1"] = (1 + beta * beta) * (vals["precision"] * vals["recall"] + eps) / \
                     (vals["precision"] * (beta * beta) + vals["recall"] + eps)

        if vals["tp"] + vals["fp"] < 10 * eps or vals["tp"] + vals["fn"] < 10 * eps or vals["precision"] + vals[
            "recall"] < 10 * eps:
            print("Possibly incorrect precision, recall or f1 values")
        return vals


def predict(model, test_loader, params):
    model.eval()
    prediction_list = []
    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample["image"].to(params["device"]).float()
            pred = model(image)
            pred = pred.cpu().numpy()
            prediction_list.append(pred)
        predictions = np.concatenate(prediction_list, axis=0)
        return predictions


def training(parameter):
    global parameters
    params = parameter
    if not os.path.isdir(params["checkpoint_dir"]):
        os.makedirs(params["checkpoint_dir"])

    train_transform = torchvision.transforms.Compose([
        ToTensorWithWeight(),
        RandomFlipWithWeight(),
        RandomRotateWithWeight()
    ])
    test_transform = torchvision.transforms.Compose([
        ToTensorWithWeight()
    ])

    train_dataset = WeightedTissueDataset(
        params["train_data_dir"],
        params["weight_dir"],
        transform=train_transform,
        channels=params["channels"],
        mode="train",
        test_id=params["test_id"],
        wrong_patient_id=params["wrong_patient_id"]
    )

    test_dataset = WeightedTissueDataset(
        params["train_data_dir"],
        params["weight_dir"],
        transform=test_transform,
        channels=params["channels"],
        mode='test',
        test_id=params["test_id"],
        wrong_patient_id=params["wrong_patient_id"]
    )

    print("train:", params["train_data_dir"], len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True,
                                               num_workers=params["workers"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False,
                                              num_workers=params["workers"])

    model = unm.UNet(in_channels=params["channels"], out_channels=2, init_features=params["init_features"])

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)
    else:
        print("Using only single GPU")

    model = model.to(params["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    train_loop(model, optimizer, train_loader, test_loader, params)
