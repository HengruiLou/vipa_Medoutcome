import torch
import tqdm
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops

from dataset.sp_hcc_dataset import HccSuperpixelsDataset, HccSuperpixelsTestDataset
from loss.fl import focal_loss
from models.gcn import GCN, GCNSimple, GraphNet
from models.graph_sage import GraphSAGE
import numpy as np

from utils.utils import _norm

dataset = HccSuperpixelsDataset(root='data/')
dataset_test = HccSuperpixelsTestDataset(root='data/')
loader = DataLoader(dataset, batch_size=1, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
hparams = {"num_node_features": 512,
           "activation": "relu",
           "num_conv_layers": 3,
           "conv_size": 128,
           "pool_method": "add",
           "lin1_size": 64,
           "lin2_size": 32,
           "output_size": 3}
device = torch.device('cuda:0')
model1 = GraphNet(c_in=768, hidden_size=512, nc=2).to(device)
model1.load_state_dict(torch.load('model_save/1.pth'))
for _, batch in tqdm.tqdm(enumerate(loader)):
    batch.x = batch.x.to(device)
    batch.edge_index = batch.edge_index.to(device)
    batch = batch.to(device)
    points = [[], []]
    pred = model1(batch)
    pred = nn.Softmax(dim=1)(pred)
    batch.y = (batch.y > 1).long()
    pred_idx = pred.argmax(dim=1).tolist()
    pred = pred.tolist()
    code = batch.code[0]
    for index, i in enumerate(pred_idx):
        if i == 1:
            p = batch.pos.tolist()[index]
            p.append(pred[index][1])
            points[0].append(p)
        else:
            p = batch.pos.tolist()[index]
            p.append(pred[index][1])
            points[1].append(p)
        # else:
        #     points[1].append(batch.pos.tolist()[index].append(pred[1]))
    np.save(f'label_store/{code}.npy', points)
