import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(config, device):
    if config.class_weights is None:
        weights = None
    else:
        weights = torch.Tensor(config.class_weights).to(device)

    if config.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=config.ignore_index, 
                                        reduction='mean', weight=weights)

    else:
        raise NotImplementedError(f"Unsupport loss type: {config.loss_type}")

    return criterion


def kd_loss_fn(config, outputs, outputsT):
    if config.kd_loss_type == 'kl_div':
        lossT = F.kl_div(F.log_softmax(outputs/config.kd_temperature, dim=1),
                    F.softmax(outputsT.detach()/config.kd_temperature, dim=1)) * config.kd_temperature ** 2

    elif config.kd_loss_type == 'mse':
        lossT = F.mse_loss(outputs, outputsT.detach())

    return lossT
