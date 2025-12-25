from multiprocessing import Pool
import os.path
import time
from itertools import repeat

import cv2
import torch
from torch import nn, tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.utils import add_self_loops
import torchmetrics
from dataset.sp_hcc_dataset import HccSuperpixelsDataset, HccSuperpixelsTestDataset
from dataset.sp_hcc_single_dataset import HccSingleSuperpixelsDataset
from loss.fl import focal_loss
from models.gcn import GCN, GCNSimple, GraphNet
from models.graph_sage import GraphSAGE
import numpy as np
import torch.nn.functional as F
from loss_schedule import loss_schedule
from models.graph_transformer import GraphTransformer, Mlp
from models.graph_transformer_pure import Transformer
from utils.utils_single import convert_numpy_img_to_superpixel_graph_2_single, \
    convert_numpy_img_to_superpixel_graph_3_single, cut_patch_stage_2_single, cut_patch_stage_3_single, \
    generate_transformer_input_and_its_groups_id_sequence, get_nodes_group_composition, merge_t_out_and_k_out, \
    multi_process_exec, read_points_from_xml, focus_nms
import logging

slide_root = '/mnt/s3/lhm/HCC/'
image_root = '/mnt/s3/lhm/HCC_seg_1/'
K = 4
writer = SummaryWriter('./log')
dataset = HccSingleSuperpixelsDataset(root='/mnt/s3/lhm/HCC_Pre/')
# dataset_test = HccSuperpixelsTestDataset(root='data/')
loader = DataLoader(dataset, batch_size=1, shuffle=True)
# loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
device = torch.device('cuda:1')
criterion = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')
model1 = GraphNet(c_in=768, hidden_size=256, nc=5).to(device)
transformer1 = Transformer(embed_dim=256, cls=2).to(device)
opt1 = torch.optim.SGD(
    [{"params": model1.parameters(), 'lr': 8e-3}, {"params": transformer1.parameters(), 'lr': 8e-3}])
scheduler1 = CosineAnnealingLR(opt1, T_max=200, eta_min=0)
final_clsfr2 = Mlp(in_features=512, hidden_features=196, out_features=2).to(device)
final_clsfr3 = Mlp(in_features=512, hidden_features=196, out_features=2).to(device)
model2 = GraphNet(c_in=768, hidden_size=256, nc=5).to(device)
graph_transformer_2 = GraphTransformer(cls=5).to(device)
opt2 = torch.optim.SGD(
    [{"params": model2.parameters(), 'lr': 3e-3}, {"params": graph_transformer_2.parameters(), 'lr': 1e-3},
     {"params": final_clsfr2.parameters(), 'lr': 3e-3}])
scheduler2 = CosineAnnealingLR(opt2, T_max=10000, eta_min=0)

model3 = GraphNet(c_in=768, hidden_size=256, nc=5).to(device)
graph_transformer_3 = GraphTransformer(cls=2).to(device)
opt3 = torch.optim.SGD(
    [{"params": model3.parameters(), 'lr': 3e-3}, {"params": graph_transformer_3.parameters(), 'lr': 1e-3},
     {"params": final_clsfr3.parameters(), 'lr': 3e-3}])
scheduler3 = CosineAnnealingLR(opt3, T_max=200, eta_min=0)

focus1 = Mlp(in_features=256, hidden_features=128, out_features=2).to(device)
opt_focus = torch.optim.SGD(focus1.parameters(), lr=3e-3, momentum=0.6)
print("Loading model state dict")
# model1.load_state_dict(torch.load('model_save/1.pth'))
# transformer1.load_state_dict(torch.load('model_save/tr1.pth'))
# model2.load_state_dict(torch.load('model_save/2.pth'))
# graph_transformer_2.load_state_dict(torch.load('model_save/gtr2.pth'))
# model3.load_state_dict(torch.load('model_save/3.pth'))
# opt_graph_transformer_3.load_state_dict(torch.load('model_save/gtr3.pth'))


print("model state dict Loaded")
train_acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=2, average=None).to(device)
train_acc2 = torchmetrics.Accuracy(task="multiclass", num_classes=2, average=None).to(device)
train_acc3 = torchmetrics.Accuracy(task="multiclass", num_classes=2, average=None).to(device)
f_acc1 = torchmetrics.Accuracy(task="multiclass", num_classes=2, average=None).to(device)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app_pure.log',  # 指定日志文件名
    filemode='w')
loss_schedule = loss_schedule(100)
for epoch in range(100):
    epoch = 1
    for img_idx, batch in enumerate(loader):
        size = np.array(batch.patch_size[0], dtype=np.int16)
        node_y = batch.node_y.to(device)
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        code = batch.code[0]
        # ===========stage1===============
        seg_map_stage_1 = np.load(f'{image_root}/{code}.npy')
        pred, feat = model1(batch)
        y1 = batch.y.long()
        y1 = y1.to(device)
        '''
        transformer 处理
        '''
        x_class_all, cls_token, x_class_node, attn, x_node = transformer1(feat)
        # for i in range(params['num_focus']):
        #     slide_name = medical_tag_path[i].split('/')[-1].split('.')[0]
        #     save_focus_map(stage_one_attention[i].cpu(), "/home/lhm/tmp/panda_test_focus/" + slide_name + '.png')
        g_attn1 = \
            torch.autograd.grad(outputs=nn.Softmax(dim=1)(x_class_all)[:, 1:5].sum(dim=(0, 1)), inputs=attn,
                                retain_graph=True)[0]
        g_attn1 = g_attn1.sum(dim=(1, 2))[:, 1:]
        g_attn1 = g_attn1.squeeze()
        x_class_node = x_class_node.squeeze()
        # x_node = x_node.detach().squeeze()
        # focus_score = focus1(x_node)
        # focus_score = F.log_softmax(focus_score, dim=1)
        y1 = (y1 > 0).long()
        loss = criterion(F.softmax(x_class_all, dim=1), y1) / 8 + criterion(F.softmax(x_class_node, dim=1), node_y) / 8
        # loss_focus = 0.7 * criterion(focus_score, node_y) + 0.3 * kl_loss(focus_score[:, 1], F.log_softmax(g_attn1))

        loss.backward()
        # loss_focus.backward()
        # if img_idx % 4 == 0:
        opt1.step()
        opt1.zero_grad()
            # opt_focus.step()
            # opt_focus.zero_grad()

        train_acc1(x_class_all, y1)
        f_acc1(x_class_node, node_y)
        if img_idx % 20 == 0:
            # logging.debug(f'{img_idx}/{len(loader)}train acc1 {train_acc1.compute()}')
            print(f'{img_idx}/{len(loader)}train acc1 {train_acc1.compute()}')
            print(f'{img_idx}/{len(loader)}focuse acc1 {f_acc1.compute()}')
        # =============prequist=============
        if loss_schedule[epoch][1][0] == 0:
            continue
        # ===========stage2===============

        selected_idx_stage_2 = focus_nms(g_attn1, batch.edge_index, 4)

        centers_stage_2 = torch.stack([batch.pos[min(i, len(batch.pos) - 1)] for i in selected_idx_stage_2])
        # if not os.path.exists(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}'):
        #     os.mkdir(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}')
        # img1 = cv2.imread(f'/nfs3/lhm/HCC/gnn_1/{code}.png')
        # for p in centers_stage_2:
        #     p = p.numpy().astype(np.int32)
        #     p[[0, 1]] = p[[1, 0]]
        #     cv2.rectangle(img1, p // 64 - 64, p // 64 + 64, color=(0, 255, 0), thickness=5)
        # for p in neg_centers_stage_2:
        #     p = p.numpy().astype(np.int32)
        #     p[[0, 1]] = p[[1, 0]]
        #     cv2.rectangle(img1, p // 64 - 64, p // 64 + 64, color=(255, 0, 0), thickness=5)
        # cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/1.png', img1)
        patches_2 = cut_patch_stage_2_single(centers_stage_2, slide_root + f'{y1.item()}/' + code + '.mrxs', size)
        # for index, p in enumerate(patches_2):
        #     cv2.imwrite(f'./2-pos-{index}.png', p)
        # for index, p in enumerate(neg_patches_2):
        #     cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/2-neg-{index}.png', p)
        # # prepare to stage3
        graphs_2 = []
        seg_maps_2 = []
        #
        # for idx, patch in enumerate(patches_2):
        #     g, seg_map = convert_numpy_img_to_superpixel_graph_2_single(patch, y1,
        #                                                                 centers_stage_2[idx] - (1024 * 8 // 2),
        #                                                                 seg_map_stage_1, batch.offset)

        #     graphs_2.append(g)
        #     seg_maps_2.append(seg_map)
        polygons = None
        if y1 > 0 and os.path.exists(slide_root + 'xml/' + f'{code}_Annotations.xml'):
            polygons = read_points_from_xml(liver_name=f'{code}_Annotations.xml', scale=64,
                                            xml_path=slide_root + 'xml/',
                                            dataset='HCC_LOWER')
        res = multi_process_exec(convert_numpy_img_to_superpixel_graph_2_single, list(
            zip(patches_2, repeat(y1.cpu().item()), centers_stage_2 - (1024 * 8 // 2), repeat(seg_map_stage_1),
                repeat(batch.offset), repeat(size), repeat(polygons))), 32)
        graphs_2, seg_maps_2 = zip(*res)

        centers_stage_3 = []
        feat_all = []
        group_ids = []
        for graph in graphs_2:
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            pred, feat = model2(graph)
            y = graph.y.long().to(device)
            # feat是gcn之后的特征
            # loss = criterion(pred, y)
            # opt2.zero_grad()
            # loss.backward()
            # opt2.step()
            # pred_y = pred.argmax(dim=1)
            # acc = [(((pred_y == y) & (y == cls)).float().sum() / (
            #         (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            # print(
            #     f'epoch {epoch}: at {index} of {len(loader)} GCN2  stage2 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            # train_acc1.append(acc)
            '''
            graph transformer 2处理
            group_ids size n//strip =(16) 本 patch的 1024个节点 对应上层 16个节点 的 ids
            '''
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            tx = tx.to(device)
            # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
            x_class_all, x_class_groups, x_class_node, cls_token_all, cls_token_groups, attn_group = graph_transformer_2(
                tx)
            cls_token_groups = cls_token_groups.squeeze()
            # pred = F.softmax(x_class_node.squeeze(0), dim=1)
            cls_out_pos = nn.Softmax(dim=1)(x_class_all)[:, 1:5]
            # g_attn_group = torch.autograd.grad(outputs=cls_out_pos.sum(dim=(0, 1)),
            #                                    inputs=attn_group, retain_graph=True)[0]
            g_attn_group = attn_group.sum(dim=(1, 2))
            g_attn_group = g_attn_group.squeeze()
            '''
            聚类处理
            '''
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
            # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
            # 中包含的维度不相符合，需要和 groups_ids对齐

            feat_final = merge_t_out_and_k_out(cls_token_groups, k_out, group_ids)
            kt_group_cls = F.softmax(final_clsfr2(feat_final), dim=1)
            '''
            指标测试
            '''
            loss = (
                           criterion(x_class_all, y)  # cls all损失
                           + criterion(kt_group_cls, y.repeat(16))  # cls group 损失
                   ) / K
            loss.backward()
            train_acc2(x_class_all, y1)
            selected_idx_stage_3 = torch.sort(g_attn_group, descending=True)[1][:K].cpu()
            pos_idx = [group_ids[i] for i in selected_idx_stage_3]

            centers_stage_3.extend([batch.pos[min(i, len(batch.pos) - 1)] for i in pos_idx])
        centers_stage_3 = torch.stack(centers_stage_3)

        if img_idx % 2 == 0:
            opt2.step()
            opt2.zero_grad()
        if img_idx % 20 == 0:
            logging.debug(f'{img_idx}/{len(loader)}train acc2 {train_acc2.compute()}')
        # ===================stage3=======================

        if loss_schedule[epoch][2][0] == 0:
            continue
        # root_3 = "/nfs3/lhm/HCC/gnn_3/"
        patches_3 = cut_patch_stage_3_single(centers_stage_3, slide_root + f'{y1.item()}/' + code + '.mrxs', size)
        # for index, p in enumerate(patches_3):
        #     cv2.imwrite(f'./3-pos-{index}.png', p)

        graphs_3 = []
        # for idx, patch in enumerate(patches_3):
        #     g = convert_numpy_img_to_superpixel_graph_3_single(patch, y1, centers_stage_3[idx] - 1024 // 2,
        #                                                     centers_stage_2[idx // K] - 1024 * 8 // 2,
        #                                                     seg_maps_2[idx // K])
        #     graphs_3.append(g)
        # patch = patch[:, :, ::-1]
        # cv2.imwrite(f'{root_3}{code}-{idx}.png', patch)
        # graphs_3 = multi_process_exec(convert_numpy_img_to_superpixel_graph_3,
        #                                   list(zip(patches_3)), 16)

        graphs_3 = multi_process_exec(
            convert_numpy_img_to_superpixel_graph_3_single,
            list(zip(patches_3, repeat(y1.cpu().item()), centers_stage_3 - (1024 // 2),
                     np.array([item for s in centers_stage_2 for item in [np.array(s)] * K]) - 1024 * 8 // 2,
                     [item for s in seg_maps_2 for item in [s] * K])), 32)

        for index, graph in enumerate(graphs_3):
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            group_y = graphs_2[index // K].y.long().to(device)
            y = graph.y.long().to(device)
            pred, feat = model3(graph)
            # loss = criterion(pred, y)
            # opt3.zero_grad()
            # loss.backward()
            # opt3.step()
            # pred_y = pred.argmax(dim=1)
            # acc = [(((pred_y == y) & (y == cls)).float().sum() / (
            #         (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            # print(
            #     f'epoch {epoch}: at {index} of {len(loader)} GCN3  stage3 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            # train_acc3.append(acc)
            '''
            graph transformer 3处理
            '''
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            tx = tx.to(device)
            # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
            x_class_all, x_class_groups, x_class_node, cls_token_all, cls_token_groups, attn_group = graph_transformer_3(
                tx)
            cls_token_groups = cls_token_groups.squeeze()
            '''
            聚类处理
            '''
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)

            # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
            # 中包含的维度不相符合，需要和 groups_ids对齐
            feat_final = merge_t_out_and_k_out(cls_token_groups, k_out, group_ids)
            kt_group_cls = F.softmax(final_clsfr3(feat_final), dim=1)
            loss = (
                           criterion(x_class_all, y1)  # cls all损失
                           + criterion(kt_group_cls, y1.repeat(16))  # cls group 损失
                   ) / (4 * K)
            loss.backward()
            train_acc3(x_class_all, y1)
            # points = []
            # for idx, i in enumerate(pred_y):
            #     if i == 1:
            #         p = graph.coord.tolist()[idx]
            #         p.append(pred[idx][1].detach())
            #         points.append(p)
            # np.save(f'{root_3}{code}-{index}.npy', points)
            # acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
            #         (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
        if img_idx % (K * K) == 0:
            opt3.step()
            opt3.zero_grad()
        if img_idx % 20 == 0:
            logging.debug(f'{img_idx}/{len(loader)}train acc3 {train_acc3.compute()}')
    logging.debug(f'epoch {epoch} total train acc1 {train_acc1.compute()}')
    logging.debug(f'epoch {epoch} total train acc2 {train_acc2.compute()}')
    logging.debug(f'epoch {epoch} total train acc3 {train_acc3.compute()}')
    train_acc1.reset()
    train_acc2.reset()
    train_acc3.reset()
    f_acc1.reset()
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    torch.save(model1.state_dict(), 'model_save/11.pth')
    torch.save(transformer1.state_dict(), 'model_save/tr11.pth')
    torch.save(focus1.state_dict(), 'model_save/focus1.pth')
    torch.save(model2.state_dict(), 'model_save/2.pth')
    torch.save(graph_transformer_2.state_dict(), 'model_save/gtr2.pth')
    torch.save(model3.state_dict(), 'model_save/3.pth')
    torch.save(graph_transformer_3.state_dict(), 'model_save/gtr3.pth')
    torch.save(final_clsfr2.state_dict(), 'model_save/fc2.pth')
    torch.save(final_clsfr3.state_dict(), 'model_save/fc3.pth')
    # model.eval()
    # for index, batch in enumerate(loader_test):
    #     pred = model(batch)
    #     pred = pred.argmax(dim=1)
    #     acc = [(((pred == batch.y) & (batch.y == cls)).float().sum() / (
    #             (batch.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
    #     test_acc.append(acc)
    # print(f'epoch {epoch} total test acc {np.mean(test_acc, axis=0)}')
    # test_acc = []
