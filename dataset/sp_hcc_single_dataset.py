import glob
import os
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data


class HccSingleSuperpixelsDataset(InMemoryDataset):

    def __init__(
            self,
            root,
            raw_root='/mnt/s3/lhm/HCC_level_1/',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            **kwargs,
    ):
        self.slic_kwargs = kwargs
        self.raw_root = raw_root
        super().__init__(os.path.join(root, "HCC_SINGLE"), transform,
                         pre_transform, pre_filter)

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

        gs = glob.glob(self.raw_root + '*.npy')
        gs_cnt = len(gs)
        # gs = gs[:int(len(gs) * 0.8)]
        for g in gs:
            graph = np.load(g, allow_pickle=True).item()
            x, node_y, edge_index, pos, y, code, offset, size = graph['x'], graph['nodey'], graph[
                'edge'], graph['pos'], graph['y'], graph['code'], graph[
                'offset'], graph['size']
            x = torch.as_tensor(x, dtype=torch.float32)
            print(x.shape)
            if len(x.shape) != 2:
                print(code)
                continue
            print(y)
            edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
            pos = torch.as_tensor(pos, dtype=torch.long)
            y = torch.as_tensor(int(y), dtype=torch.long)
            node_y = torch.as_tensor(node_y, dtype=torch.long)
            data_list.append(
                Data(x=x,
                     node_y=node_y,
                     edge_index=edge_index,
                     pos=pos,
                     y=y,
                     code=code,
                     offset=offset,
                     patch_size=size))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class HccSuperpixelsSingleTestDataset(InMemoryDataset):

    def __init__(
            self,
            root,
            raw_root='/home/lhm/mnt/medical/lhm/liverWSI/gnn_1/',
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            **kwargs,
    ):
        self.slic_kwargs = kwargs
        self.raw_root = raw_root
        super().__init__(os.path.join(root, "HCC_test"), transform,
                         pre_transform, pre_filter)

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

        gs = glob.glob(self.raw_root + '*.npy')
        gs_cnt = len(gs)
        gs = gs[int(len(gs) * 0.8):]
        for g in gs:
            graph = np.load(g, allow_pickle=True).item()
            x, edge_index, pos, y, code = graph['x'], graph['edge'], graph[
                'pos'], graph['y'], graph['code']
            x = torch.as_tensor(x, dtype=torch.float32)
            edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
            pos = torch.as_tensor(pos, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.long)
            data_list.append(
                Data(x=x, edge_index=edge_index, pos=pos, y=y, code=code))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
