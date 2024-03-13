from transform_factory import registry_transform

"""general"""
@registry_transform
def make_initial_pos(batch):
    b, device = batch['position'].shape[0], batch['position'].device
    batch['position'] = batch['position'].view(b, -1)

    return batch

"""sde"""
@registry_transform
def make_t(batch):
    bs, device = batch['position'].shape[0], batch['position'].device

    eps = 1e-3
    t = torch.rand(bs, device=device) * (1.0 - eps) + eps
    # t = torch.full((bs,), 0.1, device=device)
    batch.update(t=t)

    return batch
