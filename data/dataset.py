import torch
from transform_factory import TransformFactory

import numpy as np
import pdb

def collate_fn_r3(batch):
    def _gather(n):
        return [b[n] for b in batch]

    return dict(
        position=_gather('position')
    )

class R3Dataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.base_data = np.load(data_file)

    def _process(self, item):
        # Leave for subprocess
        return {'position': item}

    def __getitem__(self, idx):
        item = self.base_data[idx]
        ret = self._process(item)

        for k, v in ret.items():
            ret[k] = torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        return {'position': ret['position']}

class SO2Dataset(torch.utils.data.Dataset):
    def __init__(self, ):
        pass

    def __getitem__(self, idx):
        pass

class TransformedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, feats, device, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)

        self.transform_factory = TransformFactory(feats)
        self.device = device
        
    def __iter__(self,):
        for batch in super().__iter__():
            batch = {k : v.to(device=self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            yield self.transform_factory(batch)

if __name__=='__main__':
    dataset = R3Dataset(data_file='./output.npy') 
    