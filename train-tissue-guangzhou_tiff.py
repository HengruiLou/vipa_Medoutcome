from multiprocessing import Pool
import os.path
import time
from itertools import repeat
import torchmetrics
import cv2
import torch
from torch import nn, tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader

from dataset.sp_guangzhou_cell_dataset import GuangZhouCellSuperpixelsDataset
from loss.fl import focal_loss

from models.gcn import GraphNet
import numpy as np
import torch.nn.functional as F
from loss_schedule import loss_schedule
from models.graph_transformer import Mlp
from models.medical_transformer import GraphTransformer
from models.transformer import Encoder
from utils.utils_single import cluster_loss

writer = SummaryWriter('./log')
dataset = GuangZhouCellSuperpixelsDataset(root='/mnt/s3/lhm/guangzhou/pre')
loader = DataLoader(dataset, batch_size=1, shuffle=True)
device = torch.device('cuda:1')
weight = tensor([1, 0.8]).to(device)
criterion = nn.CrossEntropyLoss()
# criterion = focal_loss
clsfr = Mlp(in_features=256, hidden_features=128, out_features=13).to(device)

graph_embedding = GraphNet(c_in=768, hidden_size=256, nc=13).to(device)
# encoder = Encoder().to(device)
encoder = GraphTransformer(num_nodes=1024).to(device)
opt = torch.optim.SGD(
    [{"params": graph_embedding.parameters(), 'lr': 3e-3}, {"params": encoder.parameters(), 'lr': 1e-3},
     {"params": clsfr.parameters(), 'lr': 1e-3}])
scheduler = CosineAnnealingLR(opt, T_max=200, eta_min=0)

print("Loading model state dict")
graph_embedding.load_state_dict(torch.load('model_save/guangzhou/graph_embedding.pth'))
encoder.load_state_dict(torch.load('model_save/guangzhou/encoder.pth'))
clsfr.load_state_dict(torch.load('model_save/guangzhou/clsfr.pth'))

print("model state dict Loaded")
train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=13, average=None).to(device)
loss_schedule = loss_schedule(100)

for epoch in range(1000):
    graph_embedding.train()
    encoder.train()
    for index, batch in enumerate(loader):
        try:
            batch.x = batch.x.to(device)
            batch.edge_index = batch.edge_index.to(device)
            code = batch.code[0]
            node_num = torch.max(batch.edge_index)
            node_y = batch.node_y.long().to(device)
            logit, feat = graph_embedding(batch)
            # pred_all = F.softmax(logit, dim=1)

            '''
            transformer 处理
            '''
            padding_vector = [False] * node_num + [True] * (1024 - node_num)
            # 将这个行向量复制seq_length行，形成所需的mask矩阵
            padding_mask = torch.from_numpy(np.tile(padding_vector, (1024, 1))).to(device)
            feat = feat.unsqueeze(0)  # fake batch
            padding_mask = padding_mask.unsqueeze(0)
            # node_emb, *attn = encoder(feat, padding_mask)
            node_emb = encoder(feat, padding_mask)
            node_emb = node_emb.squeeze(0)
            pred_all = F.softmax(clsfr(node_emb), dim=1)
            pred = pred_all[:node_num, :]
            target = node_y[:node_num]
            # loss = cluster_loss(feat[:, :node_num, :], target, device)
            loss = criterion(pred, target)
            res = torch.argmax(pred, dim=1)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_acc.update(pred, target)
        except Exception as e:
            print(e)
            print(code)
    scheduler.step()
    print(train_acc.compute())
    train_acc.reset()
    torch.save(graph_embedding.state_dict(), 'model_save/guangzhou/graph_embedding.pth')
    torch.save(encoder.state_dict(), 'model_save/guangzhou/encoder.pth')
    torch.save(clsfr.state_dict(), 'model_save/guangzhou/clsfr.pth')
