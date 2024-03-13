import torch
from torch.nn as nn

def R3Loss(batch, value, cfg):
    gt_score = batch['gt_score']
    pred_score = value['pred_score']

    error = torch.sum(torch.square(pred_score - gt_score), dim=-1)
    error = torch.clip(error, 0., cfg.clamp_distance**2)

    return torch.sum(error)