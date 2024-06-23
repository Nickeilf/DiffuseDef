import os
import json
import glob
import torch

import pandas as pd

from .instance import InputInstance


def read_cls_data(fname, delimiter="\t"):
    data = pd.read_csv(fname, delimiter=delimiter, header=None)
    label_map = {}
    temp_label_set = set()

    instances = []
    for idx, row in data.iterrows():
        if len(row) == 2:
            text = row[0]
            label = row[1]
            instances.append(InputInstance(idx, text, label=label))
        elif len(row) == 3:
            # NLI task
            label = row[0]
            text_a = row[1]
            text_b = row[2]
            instances.append(InputInstance(idx, text_a, text_b=text_b, label=label))
        temp_label_set.add(label)

    for i, x in enumerate(sorted(temp_label_set)):
        label_map[x] = i

    return instances, label_map


def save_model(model, optimizer, epoch, dir, model_name):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(dir, model_name),
    )


def load_model_presume_training(model, ckpt_path, optimizer):
    weights = torch.load(ckpt_path)
    model.load_state_dict(weights["model_state_dict"])
    optimizer.load_state_dict(weights["optimizer_state_dict"])
    epoch = weights["epoch"]

    return model, optimizer, epoch


def load_model(model, ckpt_path, device):
    weights = torch.load(ckpt_path)
    model.load_state_dict(weights["model_state_dict"])
    model.to(device)

    # set eval mode
    model.eval()
    return model
