# import os.path
# import time
# from itertools import repeat
#
# import cv2
# import torch
# from torch import nn, tensor
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.data import DataLoader
# from torch_geometric.utils import add_self_loops
#
# from dataset.sp_hcc_dataset import HccSuperpixelsDataset, HccSuperpixelsTestDataset
# from loss.fl import focal_loss
# from models.gcn import GCN, GCNSimple, GraphNet
# from models.graph_sage import GraphSAGE
# import numpy as np
# import torch.nn.functional as F
# from loss_schedule import loss_schedule
# from models.graph_transformer import GraphTransformer, Mlp
# from models.graph_transformer_pure import Transformer
# from utils.utils import _norm, cut_patch_stage_2, convert_numpy_img_to_superpixel_graph_2, read_points_from_xml, \
#     cut_patch_stage_3, convert_numpy_img_to_superpixel_graph_3, multi_process_exec, multi_thread_exec, \
#     generate_transformer_input_and_its_groups_id_sequence, get_nodes_group_composition, merge_t_out_and_k_out
#
# slide_root = '/home/lhm/mnt/medical/yxt/liverWSI/Hepatoma_2_5_years_800/'
# image_root = '/home/lhm/mnt/medical/lhm/liverWSI/gnn_1_seg/'
# K = 4
# writer = SummaryWriter('./log')
# dataset = HccSuperpixelsDataset(root='data/')
# # dataset_test = HccSuperpixelsTestDataset(root='data/')
# loader = DataLoader(dataset, batch_size=1, shuffle=True)
# # loader_test = DataLoader(dataset_test, batch_size=1, shuffle=True)
# hparams = {"num_node_features": 512,
#            "activation": "relu",
#            "num_conv_layers": 3,
#            "conv_size": 128,
#            "pool_method": "add",
#            "lin1_size": 64,
#            "lin2_size": 32,
#            "output_size": 3}
# device = torch.device('cuda:1')
# weight = tensor([1, 0.8]).to(device)
# criterion = nn.CrossEntropyLoss(weight=weight)
# final_clsfr2 = Mlp(in_features=512, hidden_features=196, out_features=2).to(device)
# final_clsfr3 = Mlp(in_features=512, hidden_features=196, out_features=2).to(device)
# model1 = GraphNet(c_in=768, hidden_size=256, nc=2)
# transformer1 = Transformer().to(device)
# opt1 = torch.optim.SGD([{"params": model1.parameters(), 'lr': 3e-3}, {"params": transformer1.parameters(), 'lr': 1e-3}])
# scheduler1 = CosineAnnealingLR(opt1, T_max=200, eta_min=0)
#
# model2 = GraphNet(c_in=768, hidden_size=256, nc=2)
# graph_transformer_2 = GraphTransformer().to(device)
# opt2 = torch.optim.SGD(
#     [{"params": model2.parameters(), 'lr': 3e-3}, {"params": graph_transformer_2.parameters(), 'lr': 1e-3},
#      {"params": final_clsfr2.parameters(), 'lr': 3e-3}])
# scheduler2 = CosineAnnealingLR(opt2, T_max=200, eta_min=0)
#
# model3 = GraphNet(c_in=768, hidden_size=256, nc=2)
# graph_transformer_3 = GraphTransformer().to(device)
# opt3 = torch.optim.SGD(
#     [{"params": model3.parameters(), 'lr': 3e-3}, {"params": graph_transformer_3.parameters(), 'lr': 1e-3},
#      {"params": final_clsfr3.parameters(), 'lr': 3e-3}])
# scheduler3 = CosineAnnealingLR(opt3, T_max=200, eta_min=0)
#
# print("Loading model state dict")
# # model1.load_state_dict(torch.load('model_save/1.pth'))
# # transformer1.load_state_dict(torch.load('model_save/tr1.pth'))
# # model2.load_state_dict(torch.load('model_save/2.pth'))
# # graph_transformer_2.load_state_dict(torch.load('model_save/gtr2.pth'))
# # model3.load_state_dict(torch.load('model_save/3.pth'))
# # opt_graph_transformer_3.load_state_dict(torch.load('model_save/gtr3.pth'))
#
# model1 = model1.to(device)
# transformer1 = transformer1.to(device)
# model2 = model2.to(device)
# graph_transformer_2 = graph_transformer_2.to(device)
# model3 = model3.to(device)
# graph_transformer_3 = graph_transformer_3.to(device)
# print("model state dict Loaded")
# train_acc1 = []
# train_acc2 = []
# train_acc3 = []
# loss_schedule = loss_schedule(100)
#
# for epoch in range(100):
#     model1.train()
#     model2.train()
#     model3.train()
#     for index, batch in enumerate(loader):
#         try:
#             batch.x = batch.x.to(device)
#             batch.pos = batch.pos * 64
#             batch.edge_index = batch.edge_index.to(device)
#             code = batch.code[0]
#             # ===========stage1===============
#             seg_map_stage_1 = np.load(f'{image_root}/{code}.npy')
#             pred, feat = model1(batch)
#             y1 = batch.y.long()
#             y1 = y1.to(device)
#             # # loss = criterion(pred, y)
#             # loss = criterion(pred, y)
#             # opt1.zero_grad()
#             # loss.backward()
#             # opt1.step()
#             #
#             # pred_y = pred.argmax(dim=1)
#             # acc = [(((pred_y == y) & (y == cls)).float().sum() / (
#             #         (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
#             # print(
#             #     f'epoch {epoch}: at {index} of {len(loader)} GCN  stage1 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
#             # train_acc1.append(acc)
#             '''
#             transformer 处理
#             '''
#             if feat.shape[0] < 1024:
#                 padding_width = 1024 - feat.shape[0]
#                 # 左右上下前后
#                 feat = F.pad(feat, (0, 0, 0, padding_width), mode='constant', value=0)
#                 y1 = F.pad(y1, (0, padding_width), mode='constant', value=0)
#             x_class_all, cls_token, x_class_node, attn = transformer1(feat[:1024])
#             pred = F.softmax(x_class_node.squeeze(0), dim=1)
#             y_pad = y1[:1024]
#             # for i in range(params['num_focus']):
#             #     slide_name = medical_tag_path[i].split('/')[-1].split('.')[0]
#             #     save_focus_map(stage_one_attention[i].cpu(), "/home/lhm/tmp/panda_test_focus/" + slide_name + '.png')
#             g_attn1 = torch.autograd.grad(outputs=nn.Softmax(dim=1)(x_class_all)[:, 1].sum(),
#                                           inputs=attn, retain_graph=True)[0]
#             g_attn1 = g_attn1.sum(dim=(1, 2))[:, 1:]
#             g_attn1 = g_attn1[0]
#             loss = criterion(x_class_all.squeeze(), y_pad.long().max()) + criterion(pred, y_pad)
#             opt1.zero_grad()
#             loss.backward()
#             opt1.step()
#             pred_y = pred.argmax(dim=1)
#             acc = [(((pred_y == y_pad) & (y_pad == cls)).float().sum() / (
#                     (y_pad == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
#             print(
#                 f'epoch {epoch}: at {index} of {len(loader)} TRM stage1 acc {acc}  with gt {y_pad.sum()}/{y_pad.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
#             train_acc1.append(acc)
#
#             # =============prequist=============
#             if loss_schedule[epoch][1][0] == 0:
#                 continue
#             polygonsX8 = read_points_from_xml(liver_name=f'{code}.xml', scale=8,
#                                               xml_path=slide_root,
#                                               dataset='HCC')
#             # ===========stage2===============
#             selected_idx_stage_2 = torch.sort(g_attn1, descending=True)[1][:K].cpu()
#             centers_stage_2 = batch.pos[selected_idx_stage_2]
#             # if not os.path.exists(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}'):
#             #     os.mkdir(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}')
#             # img1 = cv2.imread(f'/nfs3/lhm/HCC/gnn_1/{code}.png')
#             # for p in centers_stage_2:
#             #     p = p.numpy().astype(np.int32)
#             #     p[[0, 1]] = p[[1, 0]]
#             #     cv2.rectangle(img1, p // 64 - 64, p // 64 + 64, color=(0, 255, 0), thickness=5)
#             # for p in neg_centers_stage_2:
#             #     p = p.numpy().astype(np.int32)
#             #     p[[0, 1]] = p[[1, 0]]
#             #     cv2.rectangle(img1, p // 64 - 64, p // 64 + 64, color=(255, 0, 0), thickness=5)
#             # cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/1.png', img1)
#             patches_2 = cut_patch_stage_2(centers_stage_2, slide_root + code + '.svs')
#             # for index, p in enumerate(patches_2):
#             #     cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/2-pos-{index}.png', p)
#             # for index, p in enumerate(neg_patches_2):
#             #     cv2.imwrite(f'/nfs3/lhm/HCC/tmp/scale1-2/{code}/2-neg-{index}.png', p)
#             # # prepare to stage3
#             # graphs_2 = []
#             # seg_maps_2 = []
#             #
#             # for idx, patch in enumerate(patches_2):
#             #     g, seg_map = convert_numpy_img_to_superpixel_graph_2(patch, polygonsX8,
#             #                                                          centers_stage_2[idx] - (1024 * 8 // 2),
#             #                                                          seg_map_stage_1)
#             #
#             #     graphs_2.append(g)
#             #     seg_maps_2.append(seg_map)
#
#             res = multi_process_exec(convert_numpy_img_to_superpixel_graph_2,
#                                      list(zip(patches_2, repeat(polygonsX8),
#                                               centers_stage_2 - (1024 * 8 // 2), repeat(seg_map_stage_1))),
#                                      8)
#             graphs_2, seg_maps_2 = zip(*res)
#
#             centers_stage_3 = []
#             accumulate_steps = len(graphs_2)
#             feat_all = []
#             group_ids = []
#             for graph in graphs_2:
#                 graph.x = graph.x.to(device)
#                 graph.edge_index = graph.edge_index.to(device)
#                 pred, feat = model2(graph)
#                 y = graph.y.long().to(device)
#                 # feat是gcn之后的特征
#                 # loss = criterion(pred, y)
#                 # opt2.zero_grad()
#                 # loss.backward()
#                 # opt2.step()
#                 # pred_y = pred.argmax(dim=1)
#                 # acc = [(((pred_y == y) & (y == cls)).float().sum() / (
#                 #         (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
#                 # print(
#                 #     f'epoch {epoch}: at {index} of {len(loader)} GCN2  stage2 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
#                 # train_acc1.append(acc)
#
#                 '''
#                 graph transformer 2处理
#                 group_ids size n//strip =(16) 本 patch的 1024个节点 对应上层 16个节点 的 ids
#             *           '''
#                 if feat.shape[0] < 1024:
#                     padding_width = 1024 - feat.shape[0]
#                     y = F.pad(y, (0, padding_width), mode='constant', value=0)
#                 tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
#                 tx = tx.to(device)
#                 # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
#                 x_class_all, x_class_groups, x_class_node, cls_token_all, cls_token_groups, attn_group = graph_transformer_2(
#                     tx)
#                 cls_token_groups = cls_token_groups.squeeze()
#                 x_class_groups = x_class_groups.squeeze()
#                 pred = F.softmax(x_class_node.squeeze(0), dim=1)
#                 y_pad = y[:1024]
#                 # g_attn = torch.autograd.grad(outputs=nn.Softmax(dim=1)(x_class_all)[:, 1].sum(),
#                 #                              inputs=attn_group, retain_graph=True)[0]
#                 attn_group = attn_group.sum(dim=(1, 2))
#                 attn_group = attn_group.squeeze()
#                 '''
#                 聚类处理
#                 '''
#                 k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
#                 # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
#                 # 中包含的维度不相符合，需要和 groups_ids对齐
#
#                 feat_final = merge_t_out_and_k_out(cls_token_groups, k_out, group_ids)
#                 kt_group_cls = F.softmax(final_clsfr2(feat_final), dim=1)
#                 '''
#                 指标测试
#                 '''
#                 group_label = y[group_ids]
#                 if len(group_label) < 16:
#                     group_label = F.pad(group_label, (0, 16 - len(group_label)), mode='constant', value=0)
#                 loss = (criterion(pred, y_pad)
#                         + criterion(x_class_all, y1[group_ids].max().unsqueeze(0))  # cls all损失
#                         + criterion(kt_group_cls, y1[group_ids])  # cls group 损失
#                         ) / accumulate_steps
#                 opt2.zero_grad()
#                 loss.backward()
#                 opt2.step()
#                 pred_y = pred.argmax(dim=1)
#                 acc = [(((pred_y == y_pad) & (y_pad == cls)).float().sum() / (
#                         (y_pad == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
#                 print(
#                     f'epoch {epoch}: at {index} of {len(loader)} TRM2 stage2 acc {acc}  with gt {y_pad.sum()}/{y_pad.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
#                 train_acc2.append(acc)
#                 selected_idx_stage_3 = torch.sort(attn_group, descending=True)[1][:K].cpu()
#                 pos_idx = [group_ids[i] for i in selected_idx_stage_3]
#                 centers_stage_3.extend([batch.pos[i] for i in pos_idx])
#
#             # ===================stage3=======================
#
#             if loss_schedule[epoch][2][0] == 0:
#                 continue
#             # root_3 = "/nfs3/lhm/HCC/gnn_3/"
#             patches_3 = cut_patch_stage_3(centers_stage_3, slide_root + code + '.svs')
#             graphs_3 = []
#             for idx, patch in enumerate(patches_3):
#                 g = convert_numpy_img_to_superpixel_graph_3(patch, centers_stage_3[idx] - 1024 // 2,
#                                                             centers_stage_2[idx // K] - 1024 * 8 // 2,
#                                                             seg_maps_2[idx // K])
#                 graphs_3.append(g)
#             # patch = patch[:, :, ::-1]
#             # cv2.imwrite(f'{root_3}{code}-{idx}.png', patch)
#             # graphs_3 = multi_process_exec(convert_numpy_img_to_superpixel_graph_3,
#             #                                   list(zip(patches_3)), 16)
#             # graphs_3 = multi_process_exec(convert_numpy_img_to_superpixel_graph_3,
#             #                          list(zip(patches_3,
#             #                                   centers_stage_3 - (1024  // 2),[item for  s in centers_stage_2 for item in [s] * K] - 1024 * 8 // 2,[item for  s in seg_maps_2 for item in [s] * K])),
#             #                          8)
#
#             accumulate_steps = len(graphs_3)
#             # loss = tensor(0, device=device)
#             for index, graph in enumerate(graphs_3):
#                 graph.x = graph.x.to(device)
#                 graph.edge_index = graph.edge_index.to(device)
#                 group_y = graphs_2[index // K].y.long().to(device)
#                 y = graph.y.long().to(device)
#                 pred, feat = model3(graph)
#                 # loss = criterion(pred, y)
#                 # opt3.zero_grad()
#                 # loss.backward()
#                 # opt3.step()
#                 # pred_y = pred.argmax(dim=1)
#                 # acc = [(((pred_y == y) & (y == cls)).float().sum() / (
#                 #         (y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
#                 # print(
#                 #     f'epoch {epoch}: at {index} of {len(loader)} GCN3  stage3 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
#                 # train_acc3.append(acc)
#                 '''
#                 graph transformer 3处理
#                 '''
#                 if feat.shape[0] < 1024:
#                     padding_width = 1024 - feat.shape[0]
#                     y = F.pad(y, (0, padding_width), mode='constant', value=0)
#                 tx, group_ids = generate_transformer_input_and_its_groups_id_sequence(graph.group_id, feat)
#                 tx = tx.to(device)
#                 # t_out为 graph transformer 的输出，维度为 16+1024，前 16位为 cls token
#                 x_class_all, x_class_groups, x_class_node, cls_token_all, cls_token_groups, attn_group = graph_transformer_3(
#                     tx)
#                 cls_token_groups = cls_token_groups.squeeze()
#                 x_class_groups = x_class_groups.squeeze()
#                 pred = F.softmax(x_class_node.squeeze(0), dim=1)
#                 y_pad = y[:1024]
#                 '''
#                 聚类处理
#                 '''
#                 k_out = get_nodes_group_composition(feat, graph.group_id, 10, device)
#
#                 # 注意，这里 cls-token中包含的 group id 最大为16个(可能不足 16个，这种情况下 cls-token是采用 padding 的方式构成 16*512的形状)，极有可能与 k_out
#                 # 中包含的维度不相符合，需要和 groups_ids对齐
#                 feat_final = merge_t_out_and_k_out(cls_token_groups, k_out, group_ids)
#                 kt_group_cls = F.softmax(final_clsfr3(feat_final), dim=1)
#                 loss = (criterion(pred, y_pad)
#                         + criterion(x_class_all, group_y[group_ids].max().unsqueeze(0))  # cls all损失
#                         + criterion(kt_group_cls, group_y[group_ids])  # cls group 损失
#                         ) / accumulate_steps
#                 opt3.zero_grad()
#                 loss.backward()
#                 opt3.step()
#                 # pred_y = pred.argmax(dim=1)
#                 pred_y = pred[:, 1] > 0.65
#                 # points = []
#                 # for idx, i in enumerate(pred_y):
#                 #     if i == 1:
#                 #         p = graph.coord.tolist()[idx]
#                 #         p.append(pred[idx][1].detach())
#                 #         points.append(p)
#                 # np.save(f'{root_3}{code}-{index}.npy', points)
#                 # acc = [(((pred_y == graph.y) & (graph.y == cls)).float().sum() / (
#                 #         (graph.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
#                 print(
#                     f'epoch {epoch}: at {index} of {len(loader)} TRM3  stage3 acc {acc}  with gt {y.sum()}/{y.shape[0]}  pred {pred_y.sum()}/{pred_y.shape[0]}')
#                 train_acc3.append(acc)
#         except Exception as e:
#             print(e)
#
#     print(f'epoch {epoch} total train acc1 {np.mean(train_acc1, axis=0)}')
#     print(f'epoch {epoch} total train acc2 {np.mean(train_acc2, axis=0)}')
#     print(f'epoch {epoch} total train acc3 {np.mean(train_acc3, axis=0)}')
#     train_acc1 = []
#     train_acc2 = []
#     train_acc3 = []
#     scheduler1.step()
#     scheduler2.step()
#     scheduler3.step()
#     torch.save(model1.state_dict(), 'model_save/1.pth')
#     torch.save(transformer1.state_dict(), 'model_save/tr1.pth')
#     torch.save(model2.state_dict(), 'model_save/2.pth')
#     torch.save(graph_transformer_2.state_dict(), 'model_save/gtr2.pth')
#     torch.save(model3.state_dict(), 'model_save/3.pth')
#     torch.save(graph_transformer_3.state_dict(), 'model_save/gtr3.pth')
#     # model.eval()
#     # for index, batch in enumerate(loader_test):
#     #     pred = model(batch)
#     #     pred = pred.argmax(dim=1)
#     #     acc = [(((pred == batch.y) & (batch.y == cls)).float().sum() / (
#     #             (batch.y == cls).float().sum() + 1e-6)).item() // 0.0001 / 100 for cls in range(2)]
#     #     test_acc.append(acc)
#     # print(f'epoch {epoch} total test acc {np.mean(test_acc, axis=0)}')
#     # test_acc = []
