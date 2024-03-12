import hydra
from omegaconf import DictConfig
import logging

from scripts.train_func_r3 import train

def setup_logger(cfg):
    pass

@hydra.main(version_base=None, config_path="config", config_name="train")   # config_name can be overwrited in bash
def main(cfg: DictConfig):
    setup_logger(cfg)

    world_type = cfg.get('world_type', 'r3')
    if world_type == 'r3':
        train(cfg)
    else:
        raise NotImplementedError(f'world {world_type} not implemented')
    
if __name__=='__main__':
    main()

