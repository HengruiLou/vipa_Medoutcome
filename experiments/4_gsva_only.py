#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# ------- 你需要安装 gseapy 和 seaborn -------
# pip install gseapy seaborn
from gseapy import gsva, get_library
import seaborn as sns

plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 12
CORE = ["geneid", "Symbol", "description"]


# ---------- 通路均值热图（带通路层次树） ----------

def plot_gsva_means_clustermap(means_df: pd.DataFrame, out_png: Path):
    """
    对通路×cluster 的均值矩阵做行聚类，画出带通路层次树的热图（类似论文 Fig.B）。
    列（C1/C2/…）不再聚类，保持 cluster 顺序不变。
    """
    sns.set(context="notebook", font="Arial", font_scale=0.8)
    g = sns.clustermap(
        means_df,
        cmap="coolwarm",
        method="ward",
        metric="euclidean",
        # 👉 把宽度从 6 提到 10，树和热图都有更多横向空间
        figsize=(10, 12),
        row_cluster=True,
        col_cluster=False,
        linewidths=0.2,
        # 👉 把行 dendrogram 的比例调大一点（默认大概是 0.2 左右）
        dendrogram_ratio=(0.3, 0.02),
        # 👉 把 colorbar 往右移，别占太多左边空间
        cbar_pos=(0.9, 0.2, 0.02, 0.5),
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_png, dpi=300)
    plt.close()


# ---------- I/O & 预处理 ----------

def read_gene_table(p: Path) -> pd.DataFrame:
    """
    读取 label_?_fpkm_top20.csv
    index = (geneid, Symbol, description)
    列 = 样本（slide_id）
    """
    df = pd.read_csv(p)
    for c in CORE:
        if c not in df.columns:
            raise RuntimeError(f"{p} 缺少列: {c}")
    df.set_index(CORE, inplace=True)
    return df


def log2_fpkm(df: pd.DataFrame) -> pd.DataFrame:
    """对 FPKM 做 log2(FPKM+1)"""
    X = df.astype(float)
    return np.log2(X + 1.0)


def align_pos_neg(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    """按 index 交集对齐正负两组，然后横向拼接。"""
    common = pos.index.intersection(neg.index)
    pos = pos.loc[common]
    neg = neg.loc[common]
    all_df = pd.concat([pos, neg], axis=1)
    return all_df


# ---------- Hallmark 基因集 ----------

def _read_gmt_simple(gmt_path: str) -> Dict[str, List[str]]:
    """简易 GMT 解析：每行 name  desc  gene1 gene2 ..."""
    gs = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                name = parts[0]
                genes = [g for g in parts[2:] if g]
                gs[name] = genes
    return gs


def load_hallmark_gene_sets(hallmark_gmt: Optional[str | Path]):
    """
    hallmark_gmt:
        - "auto"/None: 用 gseapy.get_library("Hallmark","Human") 在线获取
        - 否则视为本地 .gmt 路径
    返回 dict{name:[gene symbols]}
    """
    if hallmark_gmt is None or str(hallmark_gmt).lower() == "auto":
        print("[GSVA] Using gseapy.get_library('Hallmark','Human') ...")
        return get_library(name="Hallmark", organism="Human")
    else:
        p = Path(hallmark_gmt)
        if not p.exists():
            raise FileNotFoundError(f"Hallmark GMT not found: {p}")
        print(f"[GSVA] Loading Hallmark from {p}")
        return _read_gmt_simple(str(p))


# ---------- 为 GSVA 准备表达矩阵 ----------

def prep_expr_for_gsva(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 all_df（index=MultiIndex[geneid, Symbol, description]
              columns=sample）
    转成 expr_sym（rows=Symbol, cols=sample），同名 Symbol 取均值。
    """
    df = all_df.reset_index()
    if "Symbol" not in df.columns:
        df["Symbol"] = df["geneid"].astype(str)
    expr_sym = df.groupby("Symbol")[all_df.columns].mean()
    expr_sym = expr_sym.dropna(how="all")
    return expr_sym


# ---------- 画普通热图 ----------

def _plot_heat_matrix(data: np.ndarray,
                      xticks: List[str],
                      yticks: List[str],
                      out_png: Path,
                      title: str):
    H, W = data.shape
    fig = plt.figure(figsize=(6, max(6, H * 0.25)))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(W))
    ax.set_xticklabels(xticks, rotation=0)
    ax.set_yticks(np.arange(H))
    ax.set_yticklabels(yticks, fontsize=6)
    plt.title(title, fontweight="bold")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# ---------- 共识聚类（Consensus Clustering, KMeans 版本） ----------

def consensus_clustering_kmeans(
    X: np.ndarray,
    kmin: int = 2,
    kmax: int = 6,
    reps: int = 100,
    subsample_frac: float = 0.8,
    random_state: int = 2024
) -> Tuple[int, dict, np.ndarray, np.ndarray, dict]:
    """
    用 KMeans + 重采样做共识聚类（类似 ConsensusClusterPlus 的思想）：

    参数：
        X: (n_samples, n_features) 的样本特征矩阵（这里是 GSVA 的 pathways × samples 的转置）
        kmin, kmax: 在 [kmin, kmax] 内搜索最佳簇数 k
        reps: 每个 k 重复聚类次数
        subsample_frac: 每次重采样使用的样本比例（例如 0.8）
        random_state: 随机种子

    返回：
        best_k:      平均共识度最高的 k
        k_info:      {k: {"mean_consensus": float}}，记录每个 k 的指标
        labels:      最终用 best_k 在共识矩阵上做 KMeans 得到的簇标签（长度 = n_samples）
        consensus_best: best_k 的共识矩阵（n_samples × n_samples）
        consensus_all: {k: consensus_matrix_k}
    """
    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)

    k_info = {}
    consensus_all = {}
    best_k = None
    best_score = -np.inf

    for k in range(kmin, kmax + 1):
        print(f"[CC] Running consensus KMeans for k={k} ...")
        C = np.zeros((n_samples, n_samples), dtype=float)  # 同簇次数
        M = np.zeros((n_samples, n_samples), dtype=float)  # 被同时采样的次数

        for r in range(reps):
            # 1) 随机抽取部分样本
            n_sub = max(2, int(subsample_frac * n_samples))
            idx = rng.choice(n_samples, size=n_sub, replace=False)
            X_sub = X[idx]

            # 2) 在子样本上 KMeans 聚类
            km = KMeans(n_clusters=k, n_init=10, random_state=rng.randint(1_000_000_000))
            sub_labels = km.fit_predict(X_sub)

            # 3) 更新同簇计数 C 和共同出现计数 M
            #   同簇：只对同一簇内的样本两两加 1
            for ci in range(k):
                members = idx[sub_labels == ci]
                m = len(members)
                if m <= 1:
                    continue
                for i in range(m):
                    for j in range(i + 1, m):
                        a = members[i]
                        b = members[j]
                        C[a, b] += 1
                        C[b, a] += 1

            #   被同时采样：对子样本中的所有两两组合加 1
            for i in range(n_sub):
                for j in range(i + 1, n_sub):
                    a = idx[i]
                    b = idx[j]
                    M[a, b] += 1
                    M[b, a] += 1

        # 4) 计算共识矩阵：consensus = C / M
        with np.errstate(divide="ignore", invalid="ignore"):
            consensus = np.zeros_like(C)
            mask = M > 0
            consensus[mask] = C[mask] / M[mask]

        np.fill_diagonal(consensus, 1.0)

        # 5) 统计该 k 的平均共识度（上三角非对角元）
        tri = np.triu_indices(n_samples, k=1)
        vals = consensus[tri]
        mean_cons = float(np.nanmean(vals))
        print(f"[CC] k={k}: mean consensus = {mean_cons:.4f}")

        k_info[k] = {"mean_consensus": mean_cons}
        consensus_all[k] = consensus

        # 6) 选择平均共识度最高的 k
        if mean_cons > best_score:
            best_score = mean_cons
            best_k = k

    print(f"[CC] Best k by mean consensus = {best_k} (score={best_score:.4f})")

    # 7) 用 best_k 对对应的共识矩阵再做一次 KMeans 得到最终 labels
    consensus_best = consensus_all[best_k]
    km_final = KMeans(n_clusters=best_k, n_init=50, random_state=random_state)
    labels = km_final.fit_predict(consensus_best)

    return best_k, k_info, labels, consensus_best, consensus_all


# ---------- 主流程：GSVA + 共识聚类 ----------

def run_gsva_block(
    pos_csv: Path,
    neg_csv: Path,
    out_dir: Path,
    hallmark_gmt: Optional[str | Path],
    gsva_k: str | int = "auto"
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1. 读 FPKM & log2 变换 -----
    pos = read_gene_table(pos_csv)
    neg = read_gene_table(neg_csv)
    pos_t = log2_fpkm(pos)
    neg_t = log2_fpkm(neg)

    all_df = align_pos_neg(pos_t, neg_t)
    pos_cols = list(pos_t.columns)
    neg_cols = list(neg_t.columns)
    print(f"[GSVA] genes={all_df.shape[0]}, samples={all_df.shape[1]} (pos={len(pos_cols)}, neg={len(neg_cols)})")

    # ----- 2. 准备 Symbol 级表达矩阵 -----
    expr_sym = prep_expr_for_gsva(all_df)
    print(f"[GSVA] expr_sym: genes(Symbol)={expr_sym.shape[0]}, samples={expr_sym.shape[1]}")

    # ----- 3. 加载 Hallmark -----
    gene_sets = load_hallmark_gene_sets(hallmark_gmt)

    # ----- 4. 运行 GSVA -----
    print("[GSVA] running gsva() ...")
    gsva_res = gsva(
        data=expr_sym,
        gene_sets=gene_sets,
        no_bootstrap=True,
        sample_norm=False,
        verbose=True,
        min_size=10,
        max_size=5000,
        processes=1,
    )

    # 兼容新旧版本 gseapy
    if hasattr(gsva_res, "res2d"):
        df = gsva_res.res2d
    else:
        df = gsva_res  # 旧版可能直接是 DataFrame

    # 新版 gseapy.gsva 返回 long-format：Name, Term, ES
    if set(df.columns) >= {"Name", "Term", "ES"}:
        print("[GSVA] Detected long-format GSVA output → converting to wide matrix.")
        # index = pathway (Term), columns = sample (Name)
        gsva_scores = df.pivot(index="Term", columns="Name", values="ES")
    else:
        # 已经是 pathway × sample 的宽矩阵
        gsva_scores = df

    gsva_scores = gsva_scores.astype(float)

    gsva_dir = out_dir / "GSVA"
    gsva_dir.mkdir(parents=True, exist_ok=True)
    gsva_scores.to_csv(gsva_dir / "gsva_scores_FPKM.csv")
    print(f"[GSVA] saved gsva_scores_FPKM.csv")

    # ----- 5. 无监督共识聚类（完全不使用预后标签选 k） -----
    X = gsva_scores.T.values  # samples x pathways
    n_samples = X.shape[0]

    if isinstance(gsva_k, str) and gsva_k.lower() == "auto":
        print("[CC] gsva_k='auto' → running unsupervised consensus clustering (k=2..6) ...")
        best_k, k_info, labels, consensus_best, _ = consensus_clustering_kmeans(
            X,
            kmin=2,
            kmax=6,
            reps=100,
            subsample_frac=0.8,
            random_state=2024
        )
    else:
        k_int = int(gsva_k)
        print(f"[CC] gsva_k={k_int} → running consensus clustering with fixed k={k_int} ...")
        best_k, k_info, labels, consensus_best, _ = consensus_clustering_kmeans(
            X,
            kmin=k_int,
            kmax=k_int,
            reps=100,
            subsample_frac=0.8,
            random_state=2024
        )

    # 真实标签：前面是坏预后(1)，后面是好预后(0)，仅用于结果解释和 ARI 诊断，不参与任何聚类与选 k
    true_y = np.array([1] * len(pos_cols) + [0] * len(neg_cols), dtype=int)
    ari = adjusted_rand_score(true_y, labels) if best_k >= 2 else np.nan

    # 保存样本聚类结果（注意：cluster 是完全无监督共识聚类的结果）
    assign_df = pd.DataFrame({
        "sample": list(gsva_scores.columns),
        "gsva_cluster": labels,
        "true_label": true_y
    })
    assign_df.to_csv(gsva_dir / "gsva_assign_FPKM.csv", index=False)
    print(f"[CC] saved gsva_assign_FPKM.csv")

    # 保存 k 选择信息 + ARI（仅作诊断参考）
    with open(gsva_dir / "gsva_stats_FPKM.txt", "w") as f:
        f.write("=== Unsupervised consensus clustering (KMeans-based) ===\n")
        f.write(f"n_samples = {n_samples}\n")
        f.write("k\tmean_consensus\n")
        for k in sorted(k_info.keys()):
            f.write(f"{k}\t{k_info[k]['mean_consensus']:.6f}\n")
        f.write(f"\nChosen_k (by max mean_consensus) = {best_k}\n")
        f.write(f"ARI(true_label vs gsva_cluster) = {ari:.6f}  # for diagnostic ONLY, not used for clustering\n")
    print(f"[CC] saved gsva_stats_FPKM.txt (k selection info)")

    # 若需要，还可以把 best_k 的共识矩阵画个热图
    _plot_heat_matrix(
        consensus_best,
        xticks=list(gsva_scores.columns),
        yticks=list(gsva_scores.columns),
        out_png=gsva_dir / f"consensus_k{best_k}.png",
        title=f"Consensus matrix (k={best_k})"
    )
    print(f"[CC] saved consensus_k{best_k}.png")

    # ----- 6. 按 cluster 计算 pathway 均值（类似 Fig.B 的矩阵） -----
    means = []
    for c in range(best_k):
        means.append(gsva_scores.iloc[:, labels == c].mean(axis=1))
    means_df = pd.concat(means, axis=1)
    means_df.columns = [f"C{i+1}" for i in range(best_k)]  # C1,C2,...
    means_df.to_csv(gsva_dir / "gsva_means_FPKM.csv")
    print(f"[GSVA] saved gsva_means_FPKM.csv")

    _plot_heat_matrix(
        means_df.values,
        xticks=list(means_df.columns),
        yticks=list(means_df.index),
        out_png=gsva_dir / "gsva_means_FPKM.png",
        title=f"GSVA pathway means (k={best_k}, FPKM)"
    )
    print(f"[GSVA] saved gsva_means_FPKM.png")

    # 带通路层次树的版本（类似论文 Fig.B 左侧树）
    plot_gsva_means_clustermap(
        means_df,
        gsva_dir / "gsva_means_FPKM_clustermap.png"
    )
    print(f"[GSVA] saved gsva_means_FPKM_clustermap.png")

    # ----- 7. 样本层面的 GSVA 热图（按无监督 cluster 排序） -----
    order = np.argsort(labels)
    S_ord = gsva_scores.iloc[:, order]
    xticks = [f"{S_ord.columns[i]}(C{labels[order[i]]+1})" for i in range(S_ord.shape[1])]
    _plot_heat_matrix(
        S_ord.values,
        xticks=xticks,
        yticks=list(S_ord.index),
        out_png=gsva_dir / "gsva_samples_FPKM.png",
        title=f"GSVA scores (samples ordered by unsupervised cluster, k={best_k}, FPKM)"
    )
    print(f"[GSVA] saved gsva_samples_FPKM.png")


def main():
    ap = argparse.ArgumentParser(description="Standalone GSVA + unsupervised consensus clustering on FPKM top20 files")
    ap.add_argument("--fpkm_pos", type=Path, required=True,
                    help="label_1_fpkm_top20.csv (bad prognosis = 1)")
    ap.add_argument("--fpkm_neg", type=Path, required=True,
                    help="label_0_fpkm_top20.csv (good prognosis = 0)")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="output directory")
    ap.add_argument("--hallmark_gmt", type=str, default="auto",
                    help="'auto' or path to local Hallmark .gmt")
    ap.add_argument("--gsva_k", type=str, default="auto",
                    help="If integer (e.g. '2' or '3'): run consensus clustering with fixed k; "
                         "if 'auto': search k in [2..6] by mean consensus (unsupervised).")
    args = ap.parse_args()

    run_gsva_block(
        pos_csv=args.fpkm_pos,
        neg_csv=args.fpkm_neg,
        out_dir=args.out_dir,
        hallmark_gmt=args.hallmark_gmt,
        gsva_k=args.gsva_k,
    )


if __name__ == "__main__":
    main()
