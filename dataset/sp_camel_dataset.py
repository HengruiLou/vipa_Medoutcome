import glob
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data


class CamelyonSuperpixelsDataset(InMemoryDataset):
    def __init__(
            self,
            root,
            raw_root='/medical-data/lhm/CAMEL/lv0/tumor/',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            **kwargs,
    ):
        self.slic_kwargs = kwargs
        self.raw_root = raw_root
        super().__init__(
            os.path.join(root, "CAMEL"), transform, pre_transform, pre_filter
        )

        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return []
    # # return glob.glob(self.raw_root + '*.npy')
    #
    # @property
    # def raw_dir(self) -> str:
    #     return os.path.join(self.raw_root)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        """Dynamically generates filename for processed dataset based on superpixel parameters."""
        return ['data.pt']

    def process(self):
        data_list = []
        for g in glob.glob(self.raw_root + '*.npy'):
            graph = np.load(g, allow_pickle=True).item()
            x, edge_index, pos, y = graph['x'], graph['edge'], graph['pos'], graph['y']
            x = torch.as_tensor(x, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
            pos = torch.as_tensor(pos, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, pos=pos, y=y))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
