import hydra

@hydra.main(version_base=None, config_path="config", config_name="inference")   # config_name can be overwrited in bash
def main(cfg: DictConfig):
    setup(cfg)

    world_type = cfg.get('world_type', 'r3')
    if world_type == 'r3':
        inferece_r3(cfg)
    else:
        raise NotImplementedError(f'world {world_type} not implemented')
    
if __name__=='__main__':
    main()
