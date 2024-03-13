import torch
import logging
import numpy as np
import utils
import time

import torch
from torch import nn
from torch.optim import Adam 
from torch.utils.data.distributed import DistributedSampler

from model import NLinear, R3Diffuser
from data import R3Dataset, SO2Dataset, collate_fn_r3
from data import TransformedDataLoader as DataLoader
from loss import R3Loss


def setup_model(cfg):
    device = cfg.get('device', 'cpu')

    model_cfg = cfg.model
    if model_cfg.name == 'NLinear':
        model = NLinear(**model_cfg.NLinear)
    else:
        raise NotImplementedError(f'model {model_cfg.name} not implemented')

    if cfg.restore_model is not None:
        # if start from pretrained model
        ckpt = torch.load(cfg.restore_model)
        assert ckpt['name'] == model_cfg.name

        model.load_state_dict(ckpt['model_state_dict', strict=False])
        
    # log trainable variables
    for n, p in model.named_parameters():
        if p.requires_grad:
            logging.info(f'trainable variable {n}')

    logging.info(f'{model_cfg.name} model config')
    logging.info(model_cfg)

    # TODO: distributed training

    return model.to(device)


def setup_dataset(cfg):
    logging.info('feats: %s', cfg.transforms)

    device = cfg.get('device', 'cpu')
    
    world_type = cfg.get('world_type', 'r3')
    if world_type == 'r3':
        dataset = R3Dataset(cfg)
        collate_fn = collate_fn_r3
    elif world_type == 'so2':
        dataset = SO2Dataset(cfg)
        collate_fn = None # TODO

    # sampler = DifstributedSampler(dataset, shuffle=True, drop_last=True)  # distributed training

    train_loader = DataLoader(
        dataset = dataset,
        feats = cfg.transforms,
        device = device,
        collate_fn = collate_fn,
        batch_size = config.batch_size,
        drop_last = True
    )

    return train_loader


def train(cfg):
    utils.setup_seed(cfg.seed)

    model = setup_model(cfg)
    train_loader = setup_dataset(cfg)
    
    # optim
    tranable_variables = model.parameters()
    optim = Adam(
        tranable_variables,
        lr=cfg.lr, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0, AMSGrad=True
    )
    loss_func = R3Loss(cfg.loss)

    def _save_checkpoint(it):
        ckpt_dir = os.path.join(cfg.output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.path.makedirs(ckpt_dir)

        ckpt_file = os.path.join(ckpt_dir, f'step_{it}.ckpt')

        saved_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model # leave for distributed training

        torch.save(dict(
            name = cfg.model.name,
            model_state_dict = saved_model.state_dict(),
            optim_state_dict = optim.state_dict(),
            cfg = cfg), ckpt_file)

    model.train()
    _save_checkpoint(0)

    batch_start_time = time.time()
    cur_step = 0

    for epoch in range(cfg.num_epoch):
        # some settings

        for it, batch in enumerate(train_loader):
            jt = (it + 1) % cfg.gradient_accumulation

            ret = model(batch=batch)
            loss_info = loss_func(ret, batch)
            loss = loss_info['loss'] / cfg.gradient_accumulation

            loss.backward()

            if jt == 0:
                logging.info(f'optim step= {cur_step} lr= {cfg.lr} loss= {loss.item()}')    # TODO: Learning rate schedule
                # update grad
                torch.nn.utils.clip_grad_norm_(trainable_variables, 0.1)
                optim.step()
                optim.zero_grad()
                cur_step += 1
                # time log
                batch_end_time = time.time()
                logging.info(f'{it} batch time {batch_end_time - batch_start_time} s')
                batch_start_time = time.time()
                # save ckpt
                if cur_step % cfg.save_step == 0:
                    _save_checkpoint(cur_step)
                


