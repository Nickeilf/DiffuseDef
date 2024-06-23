from torch import nn


def get_activation_layer(activ_type: str):
    """Returns the `torch.nn.Module` for the requested activation function."""
    return {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }[activ_type.lower()]


def get_loss_function(loss_type: str):
    """Returns the `torch.nn.Module` for the requested activation function."""
    return {
        "bce": nn.BCELoss(),
        "ce": nn.CrossEntropyLoss(),
    }[loss_type.lower()]
