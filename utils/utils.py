# from __future__ import division
#
# import functools
#
# import cv2
# import torch
# import numpy as np
# import random
# import subprocess
#
# from skimage.segmentation import slic
# from torch import tensor
# from torch_geometric.data import Data
# from torch_scatter import scatter_add
# import pdb
# from torch_geometric.utils import degree, add_self_loops
# import torch.nn.functional as F
# from torch.distributions.uniform import Uniform
# import time
#
#
# def accuracy(pred, target):
#     r"""Computes the accuracy of correct predictions.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#
#     :rtype: int
#     """
#     return (pred == target).sum().item() / target.numel()
#
#
# def true_positive(pred, target, num_classes):
#     r"""Computes the number of true positive predictions.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#         num_classes (int): The number of classes.
#
#     :rtype: :class:`LongTensor`
#     """
#     out = []
#     for i in range(num_classes):
#         out.append(((pred == i) & (target == i)).sum())
#
#     return torch.tensor(out)
#
#
# def true_negative(pred, target, num_classes):
#     r"""Computes the number of true negative predictions.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#         num_classes (int): The number of classes.
#
#     :rtype: :class:`LongTensor`
#     """
#     out = []
#     for i in range(num_classes):
#         out.append(((pred != i) & (target != i)).sum())
#
#     return torch.tensor(out)
#
#
# def false_positive(pred, target, num_classes):
#     r"""Computes the number of false positive predictions.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#         num_classes (int): The number of classes.
#
#     :rtype: :class:`LongTensor`
#     """
#     out = []
#     for i in range(num_classes):
#         out.append(((pred == i) & (target != i)).sum())
#
#     return torch.tensor(out)
#
#
# def false_negative(pred, target, num_classes):
#     r"""Computes the number of false negative predictions.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#         num_classes (int): The number of classes.
#
#     :rtype: :class:`LongTensor`
#     """
#     out = []
#     for i in range(num_classes):
#         out.append(((pred != i) & (target == i)).sum())
#
#     return torch.tensor(out)
#
#
# def precision(pred, target, num_classes):
#     r"""Computes the precision:
#     :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#         num_classes (int): The number of classes.
#
#     :rtype: :class:`Tensor`
#     """
#     tp = true_positive(pred, target, num_classes).to(torch.float)
#     fp = false_positive(pred, target, num_classes).to(torch.float)
#
#     out = tp / (tp + fp)
#     out[torch.isnan(out)] = 0
#
#     return out
#
#
# def recall(pred, target, num_classes):
#     r"""Computes the recall:
#     :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#         num_classes (int): The number of classes.
#
#     :rtype: :class:`Tensor`
#     """
#     tp = true_positive(pred, target, num_classes).to(torch.float)
#     fn = false_negative(pred, target, num_classes).to(torch.float)
#
#     out = tp / (tp + fn)
#     out[torch.isnan(out)] = 0
#
#     return out
#
#
# def f1_score(pred, target, num_classes):
#     r"""Computes the :math:`F_1` score:
#     :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
#     {\mathrm{precision}+\mathrm{recall}}`.
#
#     Args:
#         pred (Tensor): The predictions.
#         target (Tensor): The targets.
#         num_classes (int): The number of classes.
#
#     :rtype: :class:`Tensor`
#     """
#     prec = precision(pred, target, num_classes)
#     rec = recall(pred, target, num_classes)
#
#     score = 2 * (prec * rec) / (prec + rec)
#     score[torch.isnan(score)] = 0
#
#     return score
#
#
# def init_seed(seed=2020):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#
#
# def get_gpu_memory_map():
#     """Get the current gpu usage.
#
#     Returns
#     -------
#     usage: dict
#         Keys are device ids as integers.
#         Values are memory usage as integers in MB.
#     """
#     result = subprocess.check_output(
#         [
#             'nvidia-smi', '--query-gpu=memory.used',
#             '--format=csv,nounits,noheader'
#         ], encoding='utf-8')
#     # Convert lines into a dictionary
#     gpu_memory = [int(x) for x in result.strip().split('\n')]
#     gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
#     return gpu_memory_map
#
#
# def _norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
#     if edge_weight is None:
#         edge_weight = torch.ones((edge_index.size(1),),
#                                  dtype=dtype,
#                                  device=edge_index.device)
#     edge_weight = edge_weight.view(-1)
#     assert edge_weight.size(0) == edge_index.size(1)
#     row, col = edge_index.detach()
#     deg = scatter_add(edge_weight.clone(), row.clone(), dim=0, dim_size=num_nodes)
#     deg_inv_sqrt = deg.pow(-1)
#     deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#
#     return deg_inv_sqrt, row, col
#
#
# # def sample_adj(edge_index, edge_weight, thr=0.5, sampling_type='random', binary=False):
# #         # tmp = (edge_weight - torch.mean(edge_weight)) / torch.std(edge_weight)
# #         if sampling_type == 'gumbel':
# #             sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1,
# #                                                                     probs=edge_weight).rsample(thr=thr)
# #         elif sampling_type == 'random':
# #             sampled = pyro.distributions.Bernoulli(1-thr).sample(edge_weight.shape).cuda()
# #         elif sampling_type == 'topk':
# #             indices = torch.topk(edge_weight, k=int(edge_weight.shape[0]*0.8))[1]
# #             sampled = torch.zeros_like(edge_weight)
# #             sampled[indices] = 1
# #         # print(sampled.sum()/edge_weight.shape[0])
# #         edge_index = edge_index[:,sampled==1]
# #         edge_weight = edge_weight*sampled
# #         edge_weight = edge_weight[edge_weight!=0]
# #         if binary:
# #             return edge_index, sampled[sampled!=0]
# #         else:
# #             return edge_index, edge_weight
#
#
# def to_heterogeneous(edge_index, num_nodes, n_id, edge_type, num_edge, device='cuda', args=None):
#     # edge_index = adj[0]
#     # num_nodes = adj[2][0]
#     edge_type_indices = []
#     # pdb.set_trace()
#     for k in range(edge_index.shape[1]):
#         edge_tmp = edge_index[:, k]
#         e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
#         edge_type_indices.append(e_type)
#     edge_type_indices = np.array(edge_type_indices)
#     A = []
#     for e_type in range(num_edge):
#         edge_tmp = edge_index[:, edge_type_indices == e_type]
#         #################################### j -> i ########################################
#         edge_tmp = torch.flip(edge_tmp, [0])
#         #################################### j -> i ########################################
#         value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
#         if args.model == 'FastGTN':
#             edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_weight=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
#             deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
#             value_tmp = deg_inv_sqrt[deg_row] * value_tmp
#         A.append((edge_tmp.to(device), value_tmp.to(device)))
#     edge_tmp = torch.stack((torch.arange(0, n_id.shape[0]), torch.arange(0, n_id.shape[0]))).type(torch.LongTensor)
#     value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
#     A.append([edge_tmp.to(device), value_tmp.to(device)])
#     return A
#
#
# # def to_heterogeneous(adj, n_id, edge_type, num_edge, device='cuda'):
# #     edge_index = adj[0]
# #     num_nodes = adj[2][0]
# #     edge_type_indices = []
# #     for k in range(edge_index.shape[1]):
# #         edge_tmp = edge_index[:,k]
# #         e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
# #         edge_type_indices.append(e_type)
# #     edge_type_indices = np.array(edge_type_indices)
# #     A = []
# #     for e_type in range(num_edge):
# #         edge_tmp = edge_index[:,edge_type_indices==e_type]
# #         value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
# #         A.append((edge_tmp.to(device), value_tmp.to(device)))
# #     edge_tmp = torch.stack((torch.arange(0,n_id.shape[0]),torch.arange(0,n_id.shape[0]))).type(torch.LongTensor)
# #     value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
# #     A.append([edge_tmp.to(device),value_tmp.to(device)])
#
# #     return A
#
#
# def generate_non_local_graph(args, feat_trans, H, A, num_edge, num_nodes):
#     K = args.K
#     # if not args.knn:
#     # pdb.set_trace()
#     x = F.relu(feat_trans(H))
#     # D_ = torch.sigmoid(x@x.t())
#     D_ = x @ x.t()
#     _, D_topk_indices = D_.t().sort(dim=1, descending=True)
#     D_topk_indices = D_topk_indices[:, :K]
#     D_topk_value = D_.t()[torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices]
#     edge_j = D_topk_indices.reshape(-1)
#     edge_i = torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K).reshape(-1).to(H.device)
#     edge_index = torch.stack([edge_i, edge_j])
#     edge_value = (D_topk_value).reshape(-1)
#     edge_value = D_topk_value.reshape(-1)
#     return [edge_index, edge_value]
#
#
# import concurrent.futures
# from tqdm import tqdm
# from multiprocessing import Pool, Pipe, freeze_support
#
# # =============================================================#
# # 接口                                                        #
# # -------------------------------------------------------------#
# #   multi_process_exec 多进程执行                             #
# #   multi_thread_exec  多线程执行                             #
# # -------------------------------------------------------------#
# # 参数：                                                      #
# #   f         (function): 批量执行的函数                      #
# #   args_mat  (list)    : 批量执行的参数                      #
# #   pool_size (int)     : 进程/线程池的大小                   #
# #   desc      (str)     : 进度条的描述文字                    #
# # -------------------------------------------------------------#
# # 例子：                                                      #
# # >>> def Pow(a,n):        ← 定义一个函数（可以有多个参数）   #
# # ...     return a**n                                         #
# # >>>                                                         #
# # >>> args_mat=[[2,1],     ← 批量计算 Pow(2,1)                #
# # ...           [2,2],                Pow(2,2)                #
# # ...           [2,3],                Pow(2,3)                #
# # ...           [2,4],                Pow(2,4)                #
# # ...           [2,5],                Pow(2,5)                #
# # ...           [2,6]]                Pow(2,6)                #
# # >>>                                                         #
# # >>> results=multi_thread_exec(Pow,args_mat,desc='计算中')   #
# # 计算中: 100%|█████████████| 6/6 [00:00<00:00, 20610.83it/s] #
# # >>>                                                         #
# # >>> print(results)                                          #
# # [2, 4, 8, 16, 32, 64]                                       #
# # -------------------------------------------------------------#
#
# ToBatch = lambda arr, size: [arr[i * size:(i + 1) * size] for i in range((size - 1 + len(arr)) // size)]
#
#
# def batch_exec(f, args_batch, w):
#     results = []
#     for i, args in enumerate(args_batch):
#         try:
#             ans = f(*args)
#             results.append(ans)
#         except Exception as e:
#             print(e)
#             results.append(None)
#         w.send(1)
#     return results
#
#
# def multi_process_exec(f, args_mat, pool_size=5, desc=None):
#     if len(args_mat) == 0: return []
#     batch_size = max(1, int(len(args_mat) / 4 / pool_size))
#     results = []
#     args_batches = ToBatch(args_mat, batch_size)
#     with tqdm(total=len(args_mat), desc=desc) as pbar:
#         with Pool(processes=pool_size) as pool:
#             r, w = Pipe(duplex=False)
#             pool_rets = []
#             for i, args_batch in enumerate(args_batches):
#                 pool_rets.append(pool.apply_async(batch_exec, (f, args_batch, w)))
#             cnt = 0
#             while cnt < len(args_mat):
#                 try:
#                     msg = r.recv()
#                     pbar.update(1)
#                     cnt += 1
#                 except EOFError:
#                     break
#             for ret in pool_rets:
#                 for r in ret.get():
#                     results.append(r)
#     return results
#
#
# def multi_thread_exec(f, args_mat, pool_size=5, desc=None):
#     if len(args_mat) == 0: return []
#     results = [None for _ in range(len(args_mat))]
#     with tqdm(total=len(args_mat), desc=desc) as pbar:
#         with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
#             futures = {executor.submit(f, *args): i for i, args in enumerate(args_mat)}
#             for future in concurrent.futures.as_completed(futures):
#                 i = futures[future]
#                 ret = future.result()
#                 results[i] = ret
#                 pbar.update(1)
#     return results
#
#
# import xml.dom.minidom as dom
#
#
# def read_points_from_xml(liver_name, scale, xml_path='/medical-data/yxt/CAMELYON16/training/lesion_annotations/',
#                          dataset=''):
#     if dataset == '':
#         if 'HCC' in xml_path:
#             dataset = 'HCC'
#         elif 'CAMELYON16' in xml_path:
#             dataset = 'CAMELYON16'
#         elif 'TCGA' in xml_path:
#             dataset = 'TCGA'
#
#     if dataset == 'HCC':
#         xml = dom.parse(xml_path + liver_name)
#         tag_name = 'Annotation'
#         point_name = 'Coordinate'
#         x_name = 'X'
#         y_name = 'Y'
#     elif dataset == 'CAMELYON16':
#         xml = dom.parse(xml_path + liver_name)
#         tag_name = 'Annotation'
#         point_name = 'Coordinate'
#         x_name = 'X'
#         y_name = 'Y'
#     elif dataset == 'TCGA':
#         xml = dom.parse(xml_path + liver_name)
#         tag_name = 'Region'
#         point_name = 'Vertex'
#         x_name = 'X'
#         y_name = 'Y'
#     anno_list = xml.documentElement.getElementsByTagName(tag_name)
#     polygons = []
#     for anno in anno_list:
#         polygons.append([])
#         for point in anno.getElementsByTagName(point_name):
#             x = int(float(point.getAttribute(x_name)))
#             y = int(float(point.getAttribute(y_name)))
#             polygons[-1].append([x / scale, y / scale])
#     return polygons
#
#
# import networkx as nx
# import numpy as np
# import torch
# from tqdm.auto import tqdm
#
#
#
# def convert_numpy_img_to_superpixel_graph_2(img, polygons, lt, pre_seg_map, slic_kwargs={}):
#     '''
#     :param img:待分割图像
#     :param polygons: 标注的多边形
#     :param lt: patch的左上角坐标（全尺寸坐标）
#     :param pre_seg_map: 上一级分割的结果，1024*1024 内容为分割node_id
#     :param slic_kwargs:
#     :return:
#     '''
#     # key = str(lt[0]) + str(lt[1])
#     # if cache1.get(key):
#     #     return cache1.get(key)
#     height = img.shape[0]
#     width = img.shape[1]
#     n = 1024
#     hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower_white = np.array([0, 0, 255 - 15])  # 假设threshold是一个你选择的值
#     upper_white = np.array([180, 255, 255])
#     mask = 255 - cv2.inRange(hsv_image, lower_white, upper_white)
#     kernel = np.ones((10, 10), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.erode(mask, kernel, iterations=1)
#     mask = mask / 255
#     segments = slic(img, n_segments=n, slic_zero=True, compactness=10, start_label=0,
#                     enforce_connectivity=True, convert2lab=True, sigma=0.7,
#                     mask=mask, **slic_kwargs)
#
#     num_of_nodes = np.max(segments) + 1
#     nodes = {node: {"rgb_list": [], "r": [], "g": [], "b": [], } for node in range(num_of_nodes)}
#
#     # get rgb values and positions
#     for y in range(height):
#         for x in range(width):
#             node = segments[y, x]
#             if node < 0:
#                 continue
#             rgb = img[y, x, :]
#             nodes[node]["r"].append(rgb[2])
#             nodes[node]["g"].append(rgb[1])
#             nodes[node]["b"].append(rgb[0])
#     for node in nodes:
#         r_bin = np.bincount(nodes[node]["r"])
#         r_bin = np.pad(r_bin, (0, 256 - len(r_bin)), 'constant', constant_values=(0, 0))
#         g_bin = np.bincount(nodes[node]["g"])
#         g_bin = np.pad(g_bin, (0, 256 - len(g_bin)), 'constant', constant_values=(0, 0))
#         b_bin = np.bincount(nodes[node]["b"])
#         b_bin = np.pad(b_bin, (0, 256 - len(b_bin)), 'constant', constant_values=(0, 0))
#         nodes[node]["rgb_list"] = np.stack([r_bin, g_bin, b_bin]).ravel()
#     G = nx.Graph()
#     # compute node positions
#     segments_ids = np.unique(segments)
#     pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
#     pos = pos.astype(int)
#     coord = pos
#     pos = pos * 8 + lt.numpy()
#     # pos[0]为height_y pos[1]为width_x
#     for node in nodes:
#         feature = nodes[node]['rgb_list']
#         label = 0
#         for p in polygons:
#             p = np.array(p, dtype=np.int32)
#             label = cv2.pointPolygonTest(p, (int(pos[node][1]) // 8, int(pos[node][0]) // 8), True) > 0
#         G.add_node(node, features=feature, label=label)
#     # add edges
#     vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
#     vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
#     bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
#     for i in range(bneighbors.shape[1]):
#         if bneighbors[0, i] == -1 or bneighbors[1, i] == -1:
#             continue
#         if bneighbors[0, i] != bneighbors[1, i]:
#             G.add_edge(bneighbors[0, i], bneighbors[1, i])
#     # add self loops
#     for node in nodes:
#         G.add_edge(node, node)
#
#     # get edge_index
#     m = len(G.edges)
#     edge_index = np.zeros([2 * m, 2]).astype(np.int64)
#     for e, (s, t) in enumerate(G.edges):
#         edge_index[e, 0] = s
#         edge_index[e, 1] = t
#         edge_index[m + e, 0] = t
#         edge_index[m + e, 1] = s
#
#     # get features
#     # num_of_nodes = len(nodes)
#     num_of_features = 768
#     x = np.zeros([num_of_nodes, num_of_features]).astype(np.float32)
#     y = np.zeros(num_of_nodes).astype(np.float32)
#     group_id = np.zeros(num_of_nodes, dtype=np.int16)
#     for node in G.nodes:
#         x[node] = G.nodes[node]["features"]
#         y[node] = G.nodes[node]["label"]
#         seg_map_point = np.array(pos[node] // 64).astype(np.int32)
#         seg_map_point[seg_map_point > 1023] = 1023
#         seg_map_point[seg_map_point < 0] = 0
#         group_id[node] = pre_seg_map[seg_map_point[0], seg_map_point[1]]
#     x = torch.as_tensor(x, dtype=torch.float32)
#     edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
#     pos = torch.as_tensor(pos, dtype=torch.float32)
#     y = torch.as_tensor(y, dtype=torch.long)
#     res = Data(x=x, edge_index=edge_index, pos=pos, y=y, coord=coord, group_id=group_id)
#     # cache1.set(key, res)
#     return res, segments
#
#
# from csbdeep.utils import normalize
#
# from stardist.models import StarDist2D
#
# star_model = StarDist2D.from_pretrained('2D_versatile_he')
#
#
# def convert_numpy_img_to_superpixel_graph_3(img,lt, stage2_lt, pre_seg_map, slic_kwargs={}):
#     # key = str(lt[0].item()) + str(lt[1].item())
#     # if cache2.get(key):
#     #     return cache2.get(key)
#     stage2_lt[stage2_lt < 0] = 0
#     stage2_lt = stage2_lt.numpy()
#     labels, _ = star_model.predict_instances(normalize(img))
#     labels[labels > 0] = 1
#     height = img.shape[0]
#     width = img.shape[1]
#     n = 1024
#     hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower_white = np.array([0, 0, 255 - 15])  # 假设threshold是一个你选择的值
#     upper_white = np.array([180, 255, 255])
#     mask = 255 - cv2.inRange(hsv_image, lower_white, upper_white)
#     kernel = np.ones((10, 10), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     mask = cv2.erode(mask, kernel, iterations=1)
#     mask = mask / 255
#     segments = slic(img, n_segments=n, slic_zero=True, compactness=10, start_label=0,
#                     enforce_connectivity=True, convert2lab=True, sigma=0.7,
#                     mask=mask, **slic_kwargs)
#     num_of_nodes = np.max(segments) + 1
#     nodes = {node: {"rgb_list": [], "r": [], "g": [], "b": [], "label_list": []} for node in range(num_of_nodes)}
#     # get rgb values and positions
#     for y in range(height):
#         for x in range(width):
#             node = segments[y, x]
#             if node < 0:
#                 continue
#             rgb = img[y, x, :]
#             nodes[node]["r"].append(rgb[2])
#             nodes[node]["g"].append(rgb[1])
#             nodes[node]["b"].append(rgb[0])
#             label = labels[y, x]
#             nodes[node]["label_list"].append(label)
#     for node in nodes:
#         r_bin = np.bincount(nodes[node]["r"])
#         r_bin = np.pad(r_bin, (0, 256 - len(r_bin)), 'constant', constant_values=(0, 0))
#         g_bin = np.bincount(nodes[node]["g"])
#         g_bin = np.pad(g_bin, (0, 256 - len(g_bin)), 'constant', constant_values=(0, 0))
#         b_bin = np.bincount(nodes[node]["b"])
#         b_bin = np.pad(b_bin, (0, 256 - len(b_bin)), 'constant', constant_values=(0, 0))
#         nodes[node]["rgb_list"] = np.stack([r_bin, g_bin, b_bin]).ravel()
#     G = nx.Graph()
#     # compute node positions
#     segments_ids = np.unique(segments)
#     pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
#     pos = pos.astype(int)
#     coord = pos
#     pos = pos + lt.numpy()
#     # pos[0]为height_y pos[1]为width_x
#     for node in nodes:
#         if node < 0:
#             continue
#         feature = nodes[node]['rgb_list']
#         label = np.sum(nodes[node]['label_list']) > 0
#         G.add_node(node, features=feature, label=label)
#     # add edges
#     vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
#     vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
#     bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
#     for i in range(bneighbors.shape[1]):
#         if bneighbors[0, i] == -1 or bneighbors[1, i] == -1:
#             continue
#         if bneighbors[0, i] != bneighbors[1, i]:
#             G.add_edge(bneighbors[0, i], bneighbors[1, i])
#     # add self loops
#     for node in nodes:
#         G.add_edge(node, node)
#
#     # get edge_index
#     m = len(G.edges)
#     edge_index = np.zeros([2 * m, 2]).astype(np.int64)
#     for e, (s, t) in enumerate(G.edges):
#         edge_index[e, 0] = s
#         edge_index[e, 1] = t
#         edge_index[m + e, 0] = t
#         edge_index[m + e, 1] = s
#
#     # get features
#     # num_of_nodes = len(nodes)
#     num_of_features = 768
#     x = np.zeros([num_of_nodes, num_of_features]).astype(np.float32)
#     y = np.zeros(num_of_nodes).astype(np.float32)
#     group_id = np.zeros(num_of_nodes)
#     for node in G.nodes:
#         x[node] = G.nodes[node]["features"]
#         y[node] = G.nodes[node]["label"]
#         seg_map_point = np.array((pos[node] - stage2_lt) // 8).astype(np.int32)
#         seg_map_point[seg_map_point > 1023] = 1023
#         seg_map_point[seg_map_point < 0] = 0
#         group_id[node] = pre_seg_map[seg_map_point[0], seg_map_point[1]]
#     x = torch.as_tensor(x, dtype=torch.float32)
#     edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
#     pos = torch.as_tensor(pos, dtype=torch.float32)
#     y = torch.as_tensor(y, dtype=torch.long)
#     res = Data(x=x, edge_index=edge_index, pos=pos, y=y, coord=coord, group_id=group_id)
#     # cache2.set(key, res)
#     return res
#
# import openslide
#
#
# def cut_patch_stage_2(centers, slide_path):
#     size = 1024
#     down_sample = 8
#     lt = centers - size * down_sample // 2
#     lt[:, [0, 1]] = lt[:, [1, 0]]
#     lt[lt < 0] = 0
#     lt = np.array(lt.tolist(), dtype=np.int32)
#     slide = openslide.open_slide(slide_path)
#     samples = np.array(slide.level_downsamples, dtype=np.int32)
#     # 1,8,32  or 1,4,16
#     res = []
#     for pt in lt:
#         if samples[1] == 4:
#             img = np.array(slide.read_region(pt.tolist(), 1, (2048, 2048)).convert('RGB'))
#             img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
#         elif samples[1] == 8:
#             img = np.array(slide.read_region(pt.tolist(), 1, (1024, 1024)).convert('RGB'))
#         res.append(img)
#     return res
#
#
# def cut_patch_stage_3(centers, slide_path):
#     size = 1024
#     centers = np.array(centers)
#     lt = centers - size // 2
#     lt[:, [0, 1]] = lt[:, [1, 0]]
#     lt[lt < 0] = 0
#     lt = np.array(lt.tolist(), dtype=np.int32)
#     slide = openslide.open_slide(slide_path)
#     res = []
#     for pt in lt:
#         img = np.array(slide.read_region(pt.tolist(), 0, (1024, 1024)).convert('RGB'))
#         res.append(img)
#     return res
#
#
# def generate_transformer_input_and_its_groups_id_sequence(groups, feat, dim=256):
#     '''
#     根据图节点的group_id数组，生成输入transformer的node_id排序结果1024=64*16，以及每一组id对应的group_id 16
#     :param groups_ids: [4,4,4,4,4,4,6,6,6,6,......]若干个节点对应的group_id，应该在随机采样至16组或padding至16
#     :return:
#     '''
#     feat = feat.cpu()
#     x = []
#     x_dict = {}
#     for idx, group_id in enumerate(groups):
#         if group_id not in x_dict:
#             x_dict[group_id] = []
#         if len(x_dict[group_id]) < 64:
#             x_dict[group_id].append(feat[idx])
#     if len(x_dict.keys()) > 16:
#         x_group = random.sample(list(x_dict.keys()), k=16)
#     else:
#         x_group = list(x_dict.keys())
#     x_group = tensor(x_group)
#     for group_id in x_group:
#         group_id = group_id.item()
#         if len(x_dict[group_id]) > 64:
#             x_dict[group_id] = x_dict[group_id][:64]
#         else:
#             while len(x_dict[group_id]) < 64:
#                 x_dict[group_id].append(torch.zeros(dim))
#         x += (x_dict[group_id])
#     x = torch.stack(x)
#     if x.shape[0] < 1024:
#         padding_width = 1024 - x.shape[0]
#         x = F.pad(x, (0, 0, 0, padding_width), mode='constant', value=0)
#         x_group = F.pad(x_group, (0, 16 - len(x_group)), mode='constant', value=0)
#     return x, x_group.long()
#
#
# from kmeans_pytorch import kmeans
#
#
# def get_nodes_group_composition(nodes_feat, group_ids, num_clusters, device):
#     '''
#     :param nodes_feat:
#     :param group_ids:
#     :param num_clusters:
#     :param device:
#     :return:
#     '''
#     assert nodes_feat.shape[0] == group_ids.shape[0]
#     cluster_ids_x, cluster_centers = kmeans(
#         X=nodes_feat, num_clusters=num_clusters, distance='euclidean', device=device
#     )
#     res = {}
#     for i in range(nodes_feat.shape[0]):
#         if group_ids[i] == -1:
#             continue
#         if group_ids[i] not in res:
#             res[group_ids[i]] = []
#         res[group_ids[i]].append(cluster_centers[cluster_ids_x[i]])
#     for key in res.keys():
#         res[key] = torch.stack(res[key]).to('cuda:1')
#         res[key] = res[key].mean(dim=0)
#     return res
#
#
# def merge_t_out_and_k_out(t_out, k_out, group_ids):
#     '''
#     将 tout 和 kout 在 groupids 包含的所有组上对齐
#     :param t_out:
#     :param k_out:
#     :param group_ids:
#     :return:
#     '''
#     feat_final = []
#     # mark 这里的 groupsids 一定是小于 16的
#     for index, group_id in enumerate(group_ids):
#         group_id = group_id.item()
#         if group_id not in k_out:
#             continue
#         feat_t = t_out[index]
#         feat_k = k_out[group_id]
#         feat_kt = torch.cat((feat_t, feat_k), dim=0)
#         feat_final.append(feat_kt)
#     if len(feat_final) == 0:
#         print('empty stack')
#     else:
#         feat_final = torch.stack(feat_final)
#     if feat_final.shape[0] < 16:
#         padding_width = 16 - feat_final.shape[0]
#         feat_final = F.pad(feat_final, (0, 0, 0, padding_width), mode='constant', value=0)
#     return feat_final
