from transform_factory import registry_transform
import torch
from model import R3Diffuser 

"""general"""


"""sde"""
@registry_transform
def make_t(batch):
    bs, device = batch['batchsize'], batch['device']

    eps = 1e-3
    t = torch.rand(bs, device=device) * (1.0 - eps) + eps
    batch.update(t=t)

    return batch

@registry_transform
def make_r3_score(batch, r3_config):
    t = batch['t']
    x_0 = batch['init_position']

    diffuser = R3Diffuser(**r3_config)
    x_t, score_t = diffuser.forward_marginal()
    batch.update(position_t=x_t, gt_score=score_t)

    return batch
