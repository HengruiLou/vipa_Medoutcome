import os.path
from itertools import repeat

import cv2
import torch
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
from loss_schedule import loss_schedule
from models.graph_transformer import GraphTransformer
from utils.utils import _norm, cut_patch_stage_2, convert_numpy_img_to_superpixel_graph_2, read_points_from_xml, \
    cut_patch_stage_3, convert_numpy_img_to_superpixel_graph_3, multi_process_exec, multi_thread_exec, \
    generate_transformer_input_and_its_groups_id_sequence, get_nodes_group_composition, merge_t_out_and_k_out

slide_root = '/home/lhm/mnt/medical/yxt/liverWSI/Hepatoma_2_5_years_800/'
image_root = '/home/lhm/mnt/medical/lhm/liverWSI/gnn_1_seg/'
K = 2
writer = SummaryWriter('./log')
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
device = torch.device('cuda:1')
criterion = nn.CrossEntropyLoss()

model1 = GraphNet(c_in=1024, hidden_size=512, nc=2).to(device)
opt1 = torch.optim.SGD(params=model1.parameters(), lr=0.001)
scheduler1 = CosineAnnealingLR(opt1, T_max=200, eta_min=0)

model2 = GraphNet(c_in=1024, hidden_size=512, nc=2).to(device)
opt2 = torch.optim.SGD(params=model2.parameters(), lr=0.001)
scheduler2 = CosineAnnealingLR(opt2, T_max=200, eta_min=0)
graph_transformer_2 = GraphTransformer()
opt_graph_transformer_2 = torch.optim.SGD(params=graph_transformer_2.parameters(), lr=0.01)

model3 = GraphNet(c_in=1024, hidden_size=512, nc=2).to(device)
opt3 = torch.optim.SGD(params=model3.parameters(), lr=0.001)
scheduler3 = CosineAnnealingLR(opt3, T_max=200, eta_min=0)
graph_transformer_3 = GraphTransformer()
opt_graph_transformer_3 = torch.optim.SGD(params=graph_transformer_3.parameters(), lr=0.01)

# model1.load_state_dict(torch.load('model_save/1.pth'))
# model2.load_state_dict(torch.load('model_save/2.pth'))
# model3.load_state_dict(torch.load('model_save/3.pth'))
train_acc1 = []
train_acc2 = []
train_acc3 = []
loss_schedule = loss_schedule(100)

for epoch in range(100):
    epoch = 60
    model1.train()
    model2.train()
    model3.train()
    for index, batch in enumerate(loader):
        batch.x = batch.x.to(device)
        batch.pos = batch.pos * 64
        batch.edge_index = batch.edge_index.to(device)
        code = batch.code[0]
        # ===========stage1===============
        seg_map_stage_1 = np.load(f'{image_root}/{code}.npy')
        pred, feat = model1(batch)
        y = (batch.y > 1).long().to(device)
        loss = criterion(pred, y)
        pred_y = pred.argmax(dim=1)
        acc = [(((pred_y == y) & (y == cls)).float().sum() / (
                (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
        print(
            f'stage1 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')

        train_acc1.append(acc)
        opt1.zero_grad()
        loss.backward()
        opt1.step()

        # =============prequist=============
        if loss_schedule[epoch][1][0] == 0:
            continue
        polygonsX8 = read_points_from_xml(liver_name=f'{code}.xml', scale=8,
                                          xml_path=slide_root,
                                          dataset='HCC')
        # ===========stage2===============
        front_bit_map = (batch.x.mean(dim=1).mean(dim=1) < 220) & (batch.x.mean(dim=1).mean(dim=1) > 0)
        front_idx = torch.arange(batch.x.shape[0]).to(device)[front_bit_map]
        pos_selected_idx_stage_2 = torch.sort(pred[front_bit_map][:, 1], descending=True)[1][:K]
        neg_selected_idx_stage_2 = torch.sort(pred[front_bit_map][:, 0], descending=True)[1][:K]
        pos_selected_idx_stage_2 = front_idx[pos_selected_idx_stage_2].cpu()
        neg_selected_idx_stage_2 = front_idx[neg_selected_idx_stage_2].cpu()
        pos_centers_stage_2 = batch.pos[pos_selected_idx_stage_2]
        neg_centers_stage_2 = batch.pos[neg_selected_idx_stage_2]
        # if not os.path.exists(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}'):
        #     os.mkdir(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}')
        # img1 = cv2.imread(f'/nfs3/lhm/HCC/gnn_1/{code}.png')
        # for p in pos_centers_stage_2:
        #     p = p.numpy().astype(np.int32)
        #     p[[0, 1]] = p[[1, 0]]
        #     cv2.rectangle(img1, p // 64 - 64, p // 64 + 64, color=(0, 255, 0), thickness=5)
        # for p in neg_centers_stage_2:
        #     p = p.numpy().astype(np.int32)
        #     p[[0, 1]] = p[[1, 0]]
        #     cv2.rectangle(img1, p // 64 - 64, p // 64 + 64, color=(255, 0, 0), thickness=5)
        # cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/1.png', img1)
        pos_patches_2 = cut_patch_stage_2(pos_centers_stage_2, slide_root + code + '.svs')
        neg_patches_2 = cut_patch_stage_2(neg_centers_stage_2, slide_root + code + '.svs')
        # for index, p in enumerate(pos_patches_2):
        #     cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/2-pos-{index}.png', p)
        # for index, p in enumerate(neg_patches_2):
        #     cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/2-neg-{index}.png', p)
        # # prepare to stage3
        pos_graphs_2 = []
        neg_graphs_2 = []
        pos_seg_maps_2 = []
        neg_seg_maps_2 = []

        for idx, patch in enumerate(pos_patches_2):
            g, seg_map = convert_numpy_img_to_superpixel_graph_2(patch, polygonsX8,
                                                                 pos_centers_stage_2[idx] - (1024 * 8 // 2),
                                                                 seg_map_stage_1)

            pos_graphs_2.append(g)
            pos_seg_maps_2.append(seg_map)

        for idx, patch in enumerate(neg_patches_2):
            g, seg_map = convert_numpy_img_to_superpixel_graph_2(patch, polygonsX8,
                                                                 neg_centers_stage_2[idx] - (1024 * 8 // 2),
                                                                 seg_map_stage_1)
            neg_graphs_2.append(g)
            neg_seg_maps_2.append(seg_map)
        # pos_graphs_2 = multi_process_exec(convert_numpy_img_to_superpixel_graph_2,
        #                                   list(zip(pos_patches_2, repeat(polygonsX8),
        #                                            pos_centers_stage_2 - (1024 * 8 // 2))),
        #                                   4)
        # neg_graphs_2 = multi_process_exec(convert_numpy_img_to_superpixel_graph_2,
        #                                   list(zip(neg_patches_2, repeat(polygonsX8),
        #                                            neg_centers_stage_2 - (1024 * 8 // 2))),
        #                                   4)
        pos_centers_stage_3 = []
        neg_centers_stage_3 = []
        accumulate_steps = len(pos_graphs_2) + len(neg_graphs_2)
        feat_all = []
        group_ids = []
        for graph in pos_graphs_2:
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            pred, feat = model2(graph)
            # feat是gcn之后的特征

            '''
            graph transformer 2处理
            '''
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
            t_out, t_attn = graph_transformer_2(tx)
            cls_tokens = t_out[:16]
            '''
            聚类处理
            '''
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
            # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
            # 中包含的维度不相符合，需要和 groups_ids对齐

            feat_final = merge_t_out_and_k_out(t_out, k_out, group_ids)
            '''
            指标测试
            '''
            graph.y = graph.y.to(device)
            loss = criterion(pred, graph.y) / accumulate_steps
            pred_y = pred.argmax(dim=1)
            acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
                    (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            print(
                f'pos stage2 acc {acc}  with gt {graph.y.sum()}/{graph.y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            train_acc2.append(acc)
            loss.backward()
            front_bit_map = (graph.x.mean(dim=1).mean(dim=1) < 220) & (graph.x.mean(dim=1).mean(dim=1) > 0)
            front_idx = torch.arange(graph.x.shape[0]).to(device)[front_bit_map]
            pos_selected_idx_stage_3 = torch.sort(pred[front_bit_map][:, 1], descending=True)[1][:K]
            pos_selected_idx_stage_3 = front_idx[pos_selected_idx_stage_3].cpu()
            pos_centers_stage_3.append(graph.pos[pos_selected_idx_stage_3])
        for graph in neg_graphs_2:
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            pred, feat = model2(graph)

            '''
            graph transformer 2处理
            '''
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
            t_out, t_attn = graph_transformer_2(tx)
            cls_tokens = t_out[:16]
            '''
            聚类处理
            '''
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
            # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
            # 中包含的维度不相符合，需要和 groups_ids对齐

            feat_final = merge_t_out_and_k_out(t_out, k_out, group_ids)
            graph.y = graph.y.to(device)
            loss = criterion(pred, graph.y) / accumulate_steps
            pred_y = pred.argmax(dim=1)
            acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
                    (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            print(
                f'neg stage2 acc {acc}  with gt {graph.y.sum()}/{graph.y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            train_acc2.append(acc)
            loss.backward()
            front_bit_map = (graph.x.mean(dim=1).mean(dim=1) < 220) & (graph.x.mean(dim=1).mean(dim=1) > 0)
            front_idx = torch.arange(graph.x.shape[0]).to(device)[front_bit_map]
            neg_selected_idx_stage_3 = torch.sort(pred[front_bit_map][:, 0], descending=True)[1][:K]
            neg_selected_idx_stage_3 = front_idx[neg_selected_idx_stage_3].cpu()
            neg_centers_stage_3.append(graph.pos[neg_selected_idx_stage_3])
        pos_centers_stage_3 = torch.cat(pos_centers_stage_3)
        neg_centers_stage_3 = torch.cat(neg_centers_stage_3)
        opt2.step()
        opt2.zero_grad()
        # ===================stage3=======================
        if loss_schedule[epoch][2][0] == 0:
            continue
        # root_3 = "/nfs3/lhm/HCC/gnn_3/"
        pos_patches_3 = cut_patch_stage_3(pos_centers_stage_3, slide_root + code + '.svs')
        neg_patches_3 = cut_patch_stage_3(neg_centers_stage_3, slide_root + code + '.svs')
        pos_graphs_3 = []
        neg_graphs_3 = []
        for idx, patch in enumerate(pos_patches_3):
            g = convert_numpy_img_to_superpixel_graph_3(patch, pos_centers_stage_3[idx] - 1024 // 2,
                                                        pos_centers_stage_2[idx // 2] - 1024 * 8 // 2,
                                                        pos_seg_maps_2[idx // 2])
            pos_graphs_3.append(g)
            # patch = patch[:, :, ::-1]
            # cv2.imwrite(f'{root_3}{code}-{idx}.png', patch)
        for idx, patch in enumerate(neg_patches_3):
            g = convert_numpy_img_to_superpixel_graph_3(patch, neg_centers_stage_3[idx] - 1024 // 2,
                                                        neg_centers_stage_2[idx // 2] - 1024 * 8 // 2,
                                                        neg_seg_maps_2[idx // 2])
            neg_graphs_3.append(g)
        # pos_graphs_3 = multi_process_exec(convert_numpy_img_to_superpixel_graph_3,
        #                                   list(zip(pos_patches_3)), 16)
        # neg_graphs_3 = multi_process_exec(convert_numpy_img_to_superpixel_graph_3,
        #                                   list(zip(neg_patches_3)), 16)
        accumulate_steps = len(pos_graphs_3) + len(neg_graphs_3)
        for index, graph in enumerate(pos_graphs_3):
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            pred, feat = model3(graph)
            '''
            graph transformer 3处理
            '''
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
            t_out, t_attn = graph_transformer_3(tx)
            cls_tokens = t_out[:16]
            '''
            聚类处理
            '''
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
            # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
            # 中包含的维度不相符合，需要和 groups_ids对齐

            feat_final = merge_t_out_and_k_out(t_out, k_out, group_ids)
            graph.y = graph.y.long().to(device)
            loss = criterion(pred, graph.y) / accumulate_steps
            # pred_y = pred.argmax(dim=1)
            pred_y = pred[:, 1] > 0.65
            # points = []
            # for idx, i in enumerate(pred_y):
            #     if i == 1:
            #         p = graph.coord.tolist()[idx]
            #         p.append(pred[idx][1].detach())
            #         points.append(p)
            # np.save(f'{root_3}{code}-{index}.npy', points)
            # acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
            #         (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            print(
                f'pos stage3 acc {acc}  with gt {graph.y.sum()}/{graph.y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            train_acc3.append(acc)
            loss.backward()
        for graph in neg_graphs_3:
            graph.x = graph.x.to(device)
            graph.edge_index = graph.edge_index.to(device)
            pred, feat = model3(graph)

            '''
            graph transformer 3处理
            '''
            tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
            # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
            t_out, t_attn = graph_transformer_3(tx)
            cls_tokens = t_out[:16]
            '''
            聚类处理
            '''
            k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
            # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
            # 中包含的维度不相符合，需要和 groups_ids对齐

            feat_final = merge_t_out_and_k_out(t_out, k_out, group_ids)
            graph.y = graph.y.long().to(device)
            loss = criterion(pred, graph.y) / accumulate_steps
            pred_y = pred.argmax(dim=1)
            acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
                    (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
            print(
                f'neg stage3 acc {acc}  with gt {graph.y.sum()}/{graph.y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
            train_acc3.append(acc)
            loss.backward()
        opt3.step()
        opt3.zero_grad()
    print(f'epoch {epoch} total train acc1 {np.mean(train_acc1, axis=0)}')
    print(f'epoch {epoch} total train acc2 {np.mean(train_acc2, axis=0)}')
    print(f'epoch {epoch} total train acc3 {np.mean(train_acc3, axis=0)}')
    train_acc1 = []
    train_acc2 = []
    train_acc3 = []
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    torch.save(model1.state_dict(), 'model_save/1.pth')
    torch.save(model2.state_dict(), 'model_save/2.pth')
    torch.save(model3.state_dict(), 'model_save/3.pth')
    # model.eval()
    # for index, batch in enumerate(loader_test):
    #     pred = model(batch)
    #     pred = pred.argmax(dim=1)
    #     acc = [(((pred == batch.y) & (batch.y == cls)).float().sum() / (
    #             (batch.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
    #     test_acc.append(acc)
    # print(f'epoch {epoch} total test acc {np.mean(test_acc, axis=0)}')
    # test_acc = []
