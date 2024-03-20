import torch
import torch.nn as nn

class R3Loss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # set some cfg
        self.cfg = cfg

    def forward(self, value, batch):
        gt_score = batch['gt_score']
        pred_score = value['pred_score']

        l2_error = torch.sum(torch.square(pred_score - gt_score), dim=-1)
        l2_error = torch.clip(l2_error, 0., self.cfg.clamp_distance**2)

        return dict(loss=torch.sum(l2_error))



