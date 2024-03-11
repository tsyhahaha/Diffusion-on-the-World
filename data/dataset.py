import torch
from transform_factory import TransformFactory

import numpy as np
import pdb

class R3Dataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.base_data = np.load(data_file)
        pdb.set_trace()

    def __getitem__(self, idx):
        return self.base_data[idx]

class TransformedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, feats, device, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

        self.transform_factory = TransformFactory(feats)
        self.device = device
        
    def set_epoch(self, epoch):
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
    
    def __iter__(self,):
        for batch in super().__iter__():
            batch = {k : v.to(device=self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            yield self.transform_factory(batch)

if __name__=='__main__':
    dataset = R3Dataset(data_file='./data/output.npy') 
    