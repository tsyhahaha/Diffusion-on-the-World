import hydra
from omegaconf import DictConfig
import logging
import os

from diffusion_world.trainer.train_func_r3 import train

def setup(cfg):
    """prepare folder, logger......"""
    os.makedirs(os.path.abspath(os.path.join(cfg.output_dir, 'checkpoints')), exist_ok=True)

    log_file = os.path.abspath(os.path.join(cfg.output_dir, 'train.log'))

    level = logging.DEBUG if cfg.verbose else logging.INFO      # Only message lower than level can be displayed
    fmt = f'%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) | %(message)s'

    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt, datefmt='%m/%d/%Y %H:%M:%S'))
        return h 
    
    handlers = [
        logging.StreamHandler(),        # write to terminal
        logging.FileHandler(log_file)   # write to 'log_file'
    ]
    handlers = list(map(_handler_apply, handlers))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers
    )

    logging.info('--------------------------')
    logging.info(f'Args: {cfg}')
    logging.info('--------------------------')


@hydra.main(version_base=None, config_path="config", config_name="train")   # config_name can be overwrited in bash
def main(cfg: DictConfig):
    setup(cfg)

    world_type = cfg.get('world_type', 'r3')
    if world_type == 'r3':
        train(cfg)
    else:
        raise NotImplementedError(f'world {world_type} not implemented')
    
if __name__=='__main__':
    main()

