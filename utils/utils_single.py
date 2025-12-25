from __future__ import division

import functools
from collections import deque

import cv2
import torch
import numpy as np
import random
import subprocess

from skimage.segmentation import slic
from torch import tensor
from torch_geometric.data import Data
from torch_scatter import scatter_add
import pdb
from torch_geometric.utils import degree, add_self_loops
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import time


def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.

    :rtype: int
    """
    return (pred == target).sum().item() / target.numel()


def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)


def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)


def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)


def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`LongTensor`
    """
    out = []
    for i in range(num_classes):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)


def precision(pred, target, num_classes):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fp = false_positive(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out


def recall(pred, target, num_classes):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes).to(torch.float)
    fn = false_negative(pred, target, num_classes).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out


def f1_score(pred, target, num_classes):
    r"""Computes the :math:`F_1` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.

    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score


def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
                                     encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def _norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)
    row, col = edge_index.detach()
    deg = scatter_add(edge_weight.clone(), row.clone(), dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    return deg_inv_sqrt, row, col


# def sample_adj(edge_index, edge_weight, thr=0.5, sampling_type='random', binary=False):
#         # tmp = (edge_weight - torch.mean(edge_weight)) / torch.std(edge_weight)
#         if sampling_type == 'gumbel':
#             sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=1,
#                                                                     probs=edge_weight).rsample(thr=thr)
#         elif sampling_type == 'random':
#             sampled = pyro.distributions.Bernoulli(1-thr).sample(edge_weight.shape).cuda()
#         elif sampling_type == 'topk':
#             indices = torch.topk(edge_weight, k=int(edge_weight.shape[0]*0.8))[1]
#             sampled = torch.zeros_like(edge_weight)
#             sampled[indices] = 1
#         # print(sampled.sum()/edge_weight.shape[0])
#         edge_index = edge_index[:,sampled==1]
#         edge_weight = edge_weight*sampled
#         edge_weight = edge_weight[edge_weight!=0]
#         if binary:
#             return edge_index, sampled[sampled!=0]
#         else:
#             return edge_index, edge_weight


def to_heterogeneous(edge_index, num_nodes, n_id, edge_type, num_edge, device='cuda', args=None):
    # edge_index = adj[0]
    # num_nodes = adj[2][0]
    edge_type_indices = []
    # pdb.set_trace()
    for k in range(edge_index.shape[1]):
        edge_tmp = edge_index[:, k]
        e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
        edge_type_indices.append(e_type)
    edge_type_indices = np.array(edge_type_indices)
    A = []
    for e_type in range(num_edge):
        edge_tmp = edge_index[:, edge_type_indices == e_type]
        #################################### j -> i ########################################
        edge_tmp = torch.flip(edge_tmp, [0])
        #################################### j -> i ########################################
        value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
        if args.model == 'FastGTN':
            edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_weight=value_tmp, fill_value=1e-20, num_nodes=num_nodes)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge_tmp.detach(), num_nodes, value_tmp.detach())
            value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp.to(device), value_tmp.to(device)))
    edge_tmp = torch.stack((torch.arange(0, n_id.shape[0]), torch.arange(0, n_id.shape[0]))).type(torch.LongTensor)
    value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
    A.append([edge_tmp.to(device), value_tmp.to(device)])
    return A


def normalize(a):
    # 最小-最大归一化
    a = a.astype(np.float32)
    a_min = a.min()
    a_max = a.max()
    normalized_a = (a - a_min) / (a_max - a_min)
    return normalized_a.astype(np.float32)


# def to_heterogeneous(adj, n_id, edge_type, num_edge, device='cuda'):
#     edge_index = adj[0]
#     num_nodes = adj[2][0]
#     edge_type_indices = []
#     for k in range(edge_index.shape[1]):
#         edge_tmp = edge_index[:,k]
#         e_type = edge_type[n_id[edge_tmp[0]].item()][n_id[edge_tmp[1]].item()]
#         edge_type_indices.append(e_type)
#     edge_type_indices = np.array(edge_type_indices)
#     A = []
#     for e_type in range(num_edge):
#         edge_tmp = edge_index[:,edge_type_indices==e_type]
#         value_tmp = torch.ones(edge_tmp.shape[1]).type(torch.FloatTensor)
#         A.append((edge_tmp.to(device), value_tmp.to(device)))
#     edge_tmp = torch.stack((torch.arange(0,n_id.shape[0]),torch.arange(0,n_id.shape[0]))).type(torch.LongTensor)
#     value_tmp = torch.ones(num_nodes).type(torch.FloatTensor)
#     A.append([edge_tmp.to(device),value_tmp.to(device)])

#     return A


def generate_non_local_graph(args, feat_trans, H, A, num_edge, num_nodes):
    K = args.K
    # if not args.knn:
    # pdb.set_trace()
    x = F.relu(feat_trans(H))
    # D_ = torch.sigmoid(x@x.t())
    D_ = x @ x.t()
    _, D_topk_indices = D_.t().sort(dim=1, descending=True)
    D_topk_indices = D_topk_indices[:, :K]
    D_topk_value = D_.t()[torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices]
    edge_j = D_topk_indices.reshape(-1)
    edge_i = torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K).reshape(-1).to(H.device)
    edge_index = torch.stack([edge_i, edge_j])
    edge_value = (D_topk_value).reshape(-1)
    edge_value = D_topk_value.reshape(-1)
    return [edge_index, edge_value]


import concurrent.futures
from tqdm import tqdm
from multiprocessing import Pool, Pipe, freeze_support

# =============================================================#
# 接口                                                        #
# -------------------------------------------------------------#
#   multi_process_exec 多进程执行                             #
#   multi_thread_exec  多线程执行                             #
# -------------------------------------------------------------#
# 参数：                                                      #
#   f         (function): 批量执行的函数                      #
#   args_mat  (list)    : 批量执行的参数                      #
#   pool_size (int)     : 进程/线程池的大小                   #
#   desc      (str)     : 进度条的描述文字                    #
# -------------------------------------------------------------#
# 例子：                                                      #
# >>> def Pow(a,n):        ← 定义一个函数（可以有多个参数）   #
# ...     return a**n                                         #
# >>>                                                         #
# >>> args_mat=[[2,1],     ← 批量计算 Pow(2,1)                #
# ...           [2,2],                Pow(2,2)                #
# ...           [2,3],                Pow(2,3)                #
# ...           [2,4],                Pow(2,4)                #
# ...           [2,5],                Pow(2,5)                #
# ...           [2,6]]                Pow(2,6)                #
# >>>                                                         #
# >>> results=multi_thread_exec(Pow,args_mat,desc='计算中')   #
# 计算中: 100%|█████████████| 6/6 [00:00<00:00, 20610.83it/s] #
# >>>                                                         #
# >>> print(results)                                          #
# [2, 4, 8, 16, 32, 64]                                       #
# -------------------------------------------------------------#

ToBatch = lambda arr, size: [arr[i * size:(i + 1) * size] for i in range((size - 1 + len(arr)) // size)]


def batch_exec(f, args_batch, w):
    results = []
    for i, args in enumerate(args_batch):
        try:
            ans = f(*args)
            results.append(ans)
        except Exception as e:
            print(e)
            results.append(None)
        w.send(1)
    return results


def multi_process_exec(f, args_mat, pool_size=5, desc=None):
    if len(args_mat) == 0: return []
    batch_size = max(1, int(len(args_mat) / 4 / pool_size))
    results = []
    args_batches = ToBatch(args_mat, batch_size)
    with tqdm(total=len(args_mat), desc=desc) as pbar:
        with Pool(processes=pool_size) as pool:
            r, w = Pipe(duplex=False)
            pool_rets = []
            for i, args_batch in enumerate(args_batches):
                pool_rets.append(pool.apply_async(batch_exec, (f, args_batch, w)))
            cnt = 0
            while cnt < len(args_mat):
                try:
                    msg = r.recv()
                    pbar.update(1)
                    cnt += 1
                except EOFError:
                    break
            for ret in pool_rets:
                for r in ret.get():
                    results.append(r)
    return results


def multi_thread_exec(f, args_mat, pool_size=5, desc=None):
    if len(args_mat) == 0: return []
    results = [None for _ in range(len(args_mat))]
    with tqdm(total=len(args_mat), desc=desc) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = {executor.submit(f, *args): i for i, args in enumerate(args_mat)}
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                ret = future.result()
                results[i] = ret
                pbar.update(1)
    return results


import xml.dom.minidom as dom


def read_points_from_xml(liver_name, scale, xml_path='/medical-data/yxt/CAMELYON16/training/lesion_annotations/',
                         dataset='', offset=None):
    if dataset == '':
        if 'HCC' in xml_path:
            dataset = 'HCC'
        elif 'CAMELYON16' in xml_path:
            dataset = 'CAMELYON16'
        elif 'TCGA' in xml_path:
            dataset = 'TCGA'

    if dataset == 'HCC':
        xml = dom.parse(xml_path + liver_name)
        tag_name = 'Annotation'
        point_name = 'Coordinate'
        x_name = 'X'
        y_name = 'Y'
    if dataset == 'HCC_LOWER':
        xml = dom.parse(xml_path + liver_name)
        tag_name = 'annotation'
        point_name = 'p'
        x_name = 'x'
        y_name = 'y'
    elif dataset == 'CAMELYON16':
        xml = dom.parse(xml_path + liver_name)
        tag_name = 'Annotation'
        point_name = 'Coordinate'
        x_name = 'X'
        y_name = 'Y'
    elif dataset == 'TCGA':
        xml = dom.parse(xml_path + liver_name)
        tag_name = 'Region'
        point_name = 'Vertex'
        x_name = 'X'
        y_name = 'Y'
    anno_list = xml.documentElement.getElementsByTagName(tag_name)
    polygons = []
    for anno in anno_list:
        polygons.append([])
        for point in anno.getElementsByTagName(point_name):
            x = int(float(point.getAttribute(x_name)))
            y = int(float(point.getAttribute(y_name)))
            # polygons[-1].append([x / scale - offset[0][1], y / scale - offset[0][0]])
            polygons[-1].append([x / scale, y / scale])
    return polygons


import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm


def convert_numpy_img_to_superpixel_graph_2_single(img, label, lt, pre_seg_map, offset, size, polygon, slic_kwargs={}):
    '''
    :param img:待分割图像
    :param polygons: 标注的多边形
    :param lt: patch的左上角坐标（全尺寸坐标）
    :param pre_seg_map: 上一级分割的结果，1024*1024 内容为分割node_id
    :param offset 切分主体区域左上角的offset  pos//64 -  offset才能对应segmap
    :param slic_kwargs:
    :return:
    '''
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
    height = img.shape[0]
    width = img.shape[1]
    n = 1024
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 220])  # 假设threshold是一个你选择的值
    upper_white = np.array([180, 255, 255])
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    # 创建掩模来分离白色和黑色背景
    mask1 = cv2.inRange(hsv_image, lower_white, upper_white)
    mask2 = cv2.inRange(hsv_image, lower_black, upper_black)
    # 使用 cv2.bitwise_or 来合并掩模
    mask = 255 - cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = mask / 255
    segments = slic(img, n_segments=n, slic_zero=True, compactness=10, start_label=0, enforce_connectivity=True,
                    convert2lab=True, sigma=0.7, mask=mask, **slic_kwargs)

    num_of_nodes = np.max(segments) + 1
    nodes = {
        node: {
            "rgb_list": [],
            "r": [],
            "g": [],
            "b": [],
        }
        for node in range(num_of_nodes)
    }

    # get rgb values and positions
    for y in range(height):
        for x in range(width):
            node = segments[y, x]
            if node < 0:
                continue
            rgb = img[y, x, :]
            nodes[node]["r"].append(rgb[2])
            nodes[node]["g"].append(rgb[1])
            nodes[node]["b"].append(rgb[0])
    for node in nodes:
        r_bin = np.bincount(nodes[node]["r"])
        r_bin = np.pad(r_bin, (0, 256 - len(r_bin)), 'constant', constant_values=(0, 0))
        g_bin = np.bincount(nodes[node]["g"])
        g_bin = np.pad(g_bin, (0, 256 - len(g_bin)), 'constant', constant_values=(0, 0))
        b_bin = np.bincount(nodes[node]["b"])
        b_bin = np.pad(b_bin, (0, 256 - len(b_bin)), 'constant', constant_values=(0, 0))
        nodes[node]["rgb_list"] = normalize(np.stack([r_bin, g_bin, b_bin]).ravel())
    G = nx.Graph()
    # compute node positions
    segments_ids = np.unique(segments)
    segments_ids = np.delete(segments_ids, np.where(segments_ids == -1))
    pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    pos = pos.astype(int)
    pos = pos * 4
    coord = pos
    pos = pos * 8 + np.array(lt)
    # pos[0]为height_y pos[1]为width_x
    for node in nodes:
        feature = nodes[node]['rgb_list']
        G.add_node(node, features=feature)
    # add edges
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] == -1 or bneighbors[1, i] == -1:
            continue
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])
    # add self loops
    for node in nodes:
        G.add_edge(node, node)

    # get edge_index
    m = len(G.edges)
    edge_index = np.zeros([2 * m, 2]).astype(np.int64)
    for e, (s, t) in enumerate(G.edges):
        edge_index[e, 0] = s
        edge_index[e, 1] = t
        edge_index[m + e, 0] = t
        edge_index[m + e, 1] = s

    # get features
    # num_of_nodes = len(nodes)
    num_of_features = 768
    x = np.zeros([1024, num_of_features]).astype(np.float32)
    group_id = np.zeros(1024, dtype=np.int16)
    for node in G.nodes:
        x[node] = G.nodes[node]["features"]
        seg_map_point = np.array((pos[node]) // 64 - offset).astype(np.int32)
        seg_map_point[0][0] = max(0, min(seg_map_point[0][0], pre_seg_map.shape[0] - 1))
        seg_map_point[0][1] = max(0, min(seg_map_point[0][1], pre_seg_map.shape[1] - 1))
        group_id[node] = pre_seg_map[seg_map_point[0][0], seg_map_point[0][1]]
    x = torch.as_tensor(x, dtype=torch.float32)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
    pos = torch.as_tensor(pos, dtype=torch.long)
    y = [label]

    if polygon:
        for p in polygon:
            p = np.array(p, dtype=np.int32)
            y = [label] if (cv2.pointPolygonTest(p, (int(lt[1] // 64), int(lt[0] // 64)), True) > 0) else [0]
    print(y)
    res = Data(x=x, edge_index=edge_index, pos=pos, y=torch.as_tensor(y), coord=coord, group_id=group_id)
    # cache1.set(key, res)
    segments = np.array(segments, dtype=np.int16)
    segments = cv2.resize(segments, size)
    return res, segments


def convert_numpy_img_to_superpixel_graph_3_single(img, label, lt, stage2_lt, pre_seg_map, slic_kwargs={}):
    img = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3))
    height = img.shape[0]
    width = img.shape[1]
    n = 1024
    # hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower_white = np.array([0, 0, 220])  # 假设threshold是一个你选择的值
    # upper_white = np.array([180, 255, 255])
    # lower_black = np.array([0, 0, 0])
    # upper_black = np.array([180, 255, 50])
    # # 创建掩模来分离白色和黑色背景
    # mask1 = cv2.inRange(hsv_image, lower_white, upper_white)
    # mask2 = cv2.inRange(hsv_image, lower_black, upper_black)
    # # 使用 cv2.bitwise_or 来合并掩模
    # mask = 255 - cv2.bitwise_or(mask1, mask2)
    # kernel = np.ones((10, 10), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # mask = cv2.erode(mask, kernel, iterations=2)
    # mask = mask / 255
    segments = slic(img, n_segments=n, slic_zero=True, compactness=10, start_label=0, enforce_connectivity=True,
                    convert2lab=True, sigma=0.7, **slic_kwargs)
    num_of_nodes = np.max(segments) + 1
    nodes = {node: {"rgb_list": [], "r": [], "g": [], "b": []} for node in range(num_of_nodes)}
    # get rgb values and positions
    for y in range(height):
        for x in range(width):
            node = segments[y, x]
            if node < 0:
                continue
            rgb = img[y, x, :]
            nodes[node]["r"].append(rgb[2])
            nodes[node]["g"].append(rgb[1])
            nodes[node]["b"].append(rgb[0])
    for node in nodes:
        r_bin = np.bincount(nodes[node]["r"])
        r_bin = np.pad(r_bin, (0, 256 - len(r_bin)), 'constant', constant_values=(0, 0))
        g_bin = np.bincount(nodes[node]["g"])
        g_bin = np.pad(g_bin, (0, 256 - len(g_bin)), 'constant', constant_values=(0, 0))
        b_bin = np.bincount(nodes[node]["b"])
        b_bin = np.pad(b_bin, (0, 256 - len(b_bin)), 'constant', constant_values=(0, 0))
        nodes[node]["rgb_list"] = normalize(np.stack([r_bin, g_bin, b_bin]).ravel())
    G = nx.Graph()
    # compute node positions
    segments_ids = np.unique(segments)
    segments_ids = np.delete(segments_ids, np.where(segments_ids == -1))
    pos = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])
    pos = pos.astype(int)
    pos = pos * 3
    coord = pos
    pos = pos + np.array(lt)
    # pos[0]为height_y pos[1]为width_x
    for node in nodes:
        feature = nodes[node]['rgb_list']
        G.add_node(node, features=feature)
    # add edges
    vs_right = np.vstack([segments[:, :-1].ravel(), segments[:, 1:].ravel()])
    vs_below = np.vstack([segments[:-1, :].ravel(), segments[1:, :].ravel()])
    bneighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1)
    for i in range(bneighbors.shape[1]):
        if bneighbors[0, i] == -1 or bneighbors[1, i] == -1:
            continue
        if bneighbors[0, i] != bneighbors[1, i]:
            G.add_edge(bneighbors[0, i], bneighbors[1, i])
    # add self loops
    for node in nodes:
        G.add_edge(node, node)

    # get edge_index
    m = len(G.edges)
    edge_index = np.zeros([2 * m, 2]).astype(np.int64)
    for e, (s, t) in enumerate(G.edges):
        edge_index[e, 0] = s
        edge_index[e, 1] = t
        edge_index[m + e, 0] = t
        edge_index[m + e, 1] = s

    # get features
    # num_of_nodes = len(nodes)
    num_of_features = 768
    stage2_lt = np.array(stage2_lt)
    x = np.zeros([1024, num_of_features]).astype(np.float32)
    group_id = np.zeros(1024)
    for node in G.nodes:
        x[node] = G.nodes[node]["features"]
        seg_map_point = np.array((pos[node] - stage2_lt) // 8).astype(np.int32)
        seg_map_point[seg_map_point > 1023] = 1023
        seg_map_point[seg_map_point < 0] = 0
        group_id[node] = pre_seg_map[seg_map_point[0], seg_map_point[1]]
    x = torch.as_tensor(x, dtype=torch.float32)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long).T
    res = Data(x=x, edge_index=edge_index, pos=pos, y=torch.as_tensor(label), coord=coord, group_id=group_id)
    # cache2.set(key, res)
    return res


import openslide


def cut_patch_stage_2_single(centers, slide_path, size):
    down_sample = 8
    size = size.astype(np.int32)
    lt = centers - size * down_sample // 2
    lt[:, [0, 1]] = lt[:, [1, 0]]
    lt[lt < 0] = 0
    lt = np.array(lt.tolist(), dtype=np.int32)
    slide = openslide.open_slide(slide_path)
    # 1,8,32  or 1,4,16
    res = []
    for pt in lt:
        img = np.array(slide.read_region(pt.tolist(), 3, (size[1], size[0])).convert('RGB'))
        res.append(img)
    return res


def cut_patch_stage_3_single(centers, slide_path, size):
    centers = np.array(centers)
    size = size.astype(np.int32)
    lt = centers - size // 2
    lt[:, [0, 1]] = lt[:, [1, 0]]
    lt[lt < 0] = 0
    lt = np.array(lt.tolist(), dtype=np.int32)
    slide = openslide.open_slide(slide_path)
    res = []
    for pt in lt:
        img = np.array(slide.read_region(pt.tolist(), 0, (size[1], size[0])).convert('RGB'))
        res.append(img)
    return res


def generate_transformer_input_and_its_groups_id_sequence(groups, feat, dim=256):
    '''
    根据图节点的group_id数组，生成输入transformer的node_id排序结果1024=64*16，以及每一组id对应的group_id 16
    :param groups_ids: [4,4,4,4,4,4,6,6,6,6,......]若干个节点对应的group_id，应该在随机采样至16组或padding至16
    :return:
    '''
    feat = feat.cpu()
    x = []
    x_dict = {}
    for idx, group_id in enumerate(groups):
        if group_id == -1:
            continue
        if group_id not in x_dict:
            x_dict[group_id] = []
        if len(x_dict[group_id]) < 64:
            x_dict[group_id].append(feat[idx])
    if len(x_dict.keys()) > 16:
        # 将key和value数组的长度存储到列表中
        lengths = [(key, len(value)) for key, value in x_dict.items()]

        # 根据数组长度降序排序
        lengths.sort(key=lambda x: x[1], reverse=True)

        # 选择前16个key
        x_group = [key for key, length in lengths[:16]]
    else:
        x_group = list(x_dict.keys())
    x_group = tensor(x_group)
    for group_id in x_group:
        group_id = group_id.item()
        if len(x_dict[group_id]) > 64:
            x_dict[group_id] = x_dict[group_id][:64]
        else:
            while len(x_dict[group_id]) < 64:
                x_dict[group_id].append(torch.zeros(dim))
        x += (x_dict[group_id])
    x = torch.stack(x)
    # 创建一个新张量，大小为 1024x512，初始值为零
    tensor_1024 = torch.zeros(1024, dim)
    x_group_16 = torch.zeros(16)
    # 复制原始张量的行到新张量中
    for i in range(x.size(0)):
        tensor_1024[i] = x[i]
    for i in range(x_group.size(0)):
        x_group_16[i] = x_group[i]
    if x.shape[0] < 1024:
        # 复制最后一行来填充剩余的高度
        for i in range(x.size(0), tensor_1024.size(0)):
            tensor_1024[i] = x[-1]
        for i in range(x_group.size(0), x_group_16.size(0)):
            x_group_16[i] = x_group[-1]
    return tensor_1024, x_group_16.long()


from torch_kmeans import KMeans

model = KMeans(n_clusters=10).to('cuda:1')


def get_nodes_group_composition(nodes_feat, group_ids, num_clusters, device):
    '''
    :param nodes_feat:
    :param group_ids:
    :param num_clusters:
    :param device:
    :return:
    '''
    assert nodes_feat.shape[0] == group_ids.shape[0]
    nodes_feat = nodes_feat.unsqueeze(0)
    res = model(nodes_feat)
    cluster_centers = res.centers.squeeze()
    cluster_ids_x = res.labels.squeeze()
    res = {}
    for i in range(group_ids.shape[0]):
        if group_ids[i] == -1:
            continue
        if group_ids[i] not in res:
            res[group_ids[i]] = []
        res[group_ids[i]].append(cluster_centers[cluster_ids_x[i]])
    for key in res.keys():
        res[key] = torch.stack(res[key]).to('cuda:1')
        res[key] = res[key].mean(dim=0)
    return res


def merge_t_out_and_k_out(t_out, k_out, group_ids):
    '''
    将 tout 和 kout 在 groupids 包含的所有组上对齐
    :param t_out:
    :param k_out:
    :param group_ids:
    :return:
    '''
    feat_final = []
    # mark 这里的 groupsids 一定是小于 16的
    for index, group_id in enumerate(group_ids):
        group_id = group_id.item()
        if group_id not in k_out:
            continue
        feat_t = t_out[index]
        feat_k = k_out[group_id]
        feat_kt = torch.cat((feat_t, feat_k), dim=0)
        feat_final.append(feat_kt)
    if len(feat_final) == 0:
        print('empty stack')
        feat_final = torch.tensor(feat_final)
    else:
        feat_final = torch.stack(feat_final)
    if feat_final.shape[0] < 16:
        padding_width = 16 - feat_final.shape[0]
        feat_final = F.pad(feat_final, (0, 0, 0, padding_width), mode='constant', value=0)
    return feat_final


def draw_graph_attn(attn, segments):
    '''
    :param attn: attn序列，长度1024 索引为 segments 的节点 id
    :param segments: 原图对应节点 id
    :return:
    '''
    attn = attn.detach().cpu().numpy()
    attn_score = np.full(segments.shape, -np.inf)
    # attn = normalize(attn)
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            if segments[i][j] == -1:
                continue
            attn_score[i, j] = attn[segments[i][j]]
    return attn_score


def draw_graph_labels(polygon, segments):
    '''
    :param polygon: polygon序列，长度1024 索引为 segments 的节点 id
    :param segments: 原图对应节点 id
    :return:
    '''
    polygon = polygon.detach().cpu().numpy()
    labels_score = np.zeros(segments.shape)
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            labels_score[i, j] = labels[segments[i][j]]
    return labels


def focus_nms(g_attn, edges, K):
    focus_res = []
    edges = edges.detach().cpu().numpy().transpose(1, 0)
    adjacency_list = {i: [] for i in range(len(g_attn))}
    for (node1_index, node2_index) in edges:
        adjacency_list[node1_index].append(node2_index)
        adjacency_list[node2_index].append(node1_index)

    def bfs_layers(start_node_index, max_depth=5):
        # 检查起始节点索引是否有效
        if start_node_index < 0 or start_node_index >= len(g_attn):
            return "Invalid start node index."

        # 创建队列并初始化访问状态
        queue = deque([(start_node_index, 0)])  # 元组包含节点索引和当前层级
        visited = set()
        visited.add(start_node_index)

        while queue:
            current_node_index, current_depth = queue.popleft()

            # 如果达到最大深度，停止搜索
            if current_depth >= max_depth:
                break

            # 遍历当前节点的所有邻居节点
            for neighbor_index in adjacency_list[current_node_index.item()]:
                if neighbor_index not in visited:
                    visited.add(neighbor_index)
                    queue.append((neighbor_index, current_depth + 1))
                    # 更新BFS结果
                    if (neighbor_index < len(g_attn) and neighbor_index > -1):
                        g_attn[neighbor_index] = -torch.inf

    for i in range(K):
        sorted_idx = torch.sort(g_attn, descending=True)[1].cpu()
        focus_res.append(sorted_idx[0])
        bfs_layers(sorted_idx[0], 6)
    return focus_res


def cluster_loss(feat, y, device):
    labels = torch.unique(y)
    feat = feat.squeeze()
    feat = F.normalize(feat, p=2, dim=1)
    n, d = feat.shape
    mask = torch.full((n, n), -1).to(device)
    for l in labels:
        idxs = torch.nonzero(y == l).squeeze()
        if len(idxs.shape) == 0:
            continue
        coords = torch.combinations(idxs, 2).t()
        mask[coords[0],coords[1]] = 1
    indices = torch.arange(n)
    mask[indices, indices] = 0
    sim = feat @ feat.T
    loss_matrix = sim * mask
    loss_all = -torch.sum(loss_matrix) / n
    return torch.exp(loss_all)
