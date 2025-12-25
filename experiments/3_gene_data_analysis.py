#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, optimal_leaf_ordering, leaves_list
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter

# ---- Matplotlib global (Arial + ~15pt for PPT)
plt.rcParams['font.family'] = ['Arial']      # enforce Arial
plt.rcParams['font.size'] = 15               # base font size ~ PPT 15pt
plt.rcParams['axes.unicode_minus'] = False

np.set_printoptions(suppress=True)
CORE = ["geneid", "Symbol", "description"]


# ---------------- Helpers ----------------
def pnames_from(names: List[str]) -> List[str]:
    """Return P1..PN for given list length; order preserved."""
    return [f"P{i+1}" for i in range(len(names))]


# === NEW: Abbreviate Ensembl gene IDs for display ============================
def abbreviate_geneid(gid: str) -> str:
    """
    Abbreviate Ensembl gene IDs for display:
    - Remove version suffix (e.g., '.12')
    - If prefix 'ENSG00000' is present, drop it and keep the last 6 digits
    - If 'ENSG' but not 'ENSG00000', keep last 6 digits of numeric part if possible
    - Otherwise, fallback to last 6 alphanumerics
    """
    s = str(gid).split('.', 1)[0]
    if s.startswith("ENSG00000"):
        tail = s[len("ENSG00000"):]
        return tail[-6:] if len(tail) > 6 else tail
    elif s.startswith("ENSG"):
        digits = "".join(ch for ch in s if ch.isdigit())
        return digits[-6:] if digits else s[-6:]
    else:
        import re
        alnum = re.findall(r"[A-Za-z0-9]", s)
        return "".join(alnum[-6:]) if alnum else s[-6:]


# ---------------- I/O & transforms ----------------
def read_gene_table(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p)
    for c in CORE:
        if c not in df.columns:
            raise RuntimeError(f"{p} is missing required column: {c}")
    df.set_index(CORE, inplace=True)
    return df


def log2_fpkm(df: pd.DataFrame) -> pd.DataFrame:
    X = df.astype(float)
    return np.log2(X + 1.0)


def log2_cpm(count_df: pd.DataFrame) -> pd.DataFrame:
    X = count_df.astype(float)
    libsize = X.sum(axis=0) + 1e-8
    cpm = (X / libsize) * 1e6
    return np.log2(cpm + 1.0)


def align_pos_neg(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    common = pos.index.intersection(neg.index)
    pos = pos.loc[common]
    neg = neg.loc[common]
    all_df = pd.concat([pos, neg], axis=1)
    return all_df


def build_label_vector(pos_cols, neg_cols):
    # pos (bad=1) first, then neg (good=0)
    y = np.array([1]*len(pos_cols) + [0]*len(neg_cols), dtype=int)
    return y


# ---------------- DEG & selection ----------------
def welch_deg(all_df: pd.DataFrame, pos_cols, neg_cols):
    pvals, l2fc = [], []
    for _, row in all_df.iterrows():
        a = row[pos_cols].values.astype(float)
        b = row[neg_cols].values.astype(float)
        t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        if np.isnan(p):
            p = 1.0
        pvals.append(p)
        l2fc.append(np.nanmean(a) - np.nanmean(b))
    pvals = np.array(pvals)
    l2fc = np.array(l2fc)
    _, qvals, *_ = multipletests(pvals, alpha=0.05, method='fdr_bh')
    res = pd.DataFrame({"p_value": pvals, "q_value": qvals, "log2FC": l2fc}, index=all_df.index)
    return res


def choose_de_genes(res: pd.DataFrame, p_th=0.05, l2_th=1.0, top=100):
    sig = res[(res["p_value"] < p_th) & (np.abs(res["log2FC"]) > l2_th)]
    if len(sig) == 0:
        sig = res.sort_values("p_value").head(min(top, len(res)))
    elif len(sig) > top:
        sig = sig.sort_values("p_value").head(top)
    return sig.index


# ---------------- Pretty volcano (color by quadrant; no labels/legend) ----------------
def plot_volcano_pretty(res: pd.DataFrame, out_png: Path,
                        p_thr=0.05, fc_thr=1.0, title="Volcano"):
    x = res["log2FC"].values
    y = -np.log10(np.clip(res["p_value"].values, 1e-300, None))
    ythr = -np.log10(p_thr)

    # masks
    top_right = (x >= fc_thr) & (y >= ythr)
    top_left  = (x <= -fc_thr) & (y >= ythr)
    sig_other = (~top_right) & (~top_left)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')

    # Set symmetric limits BEFORE shading
    xmax = float(np.nanmax(np.abs(x))) if len(x) else 1.0
    ymax = float(np.nanmax(y)) if len(y) else 1.0
    xmax = max(xmax, fc_thr) * 1.05
    ymax = max(ymax, ythr) * 1.05
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, ymax)

    # thresholds
    ax.axvline(+fc_thr, ls='--', lw=1.2, c='#555555')
    ax.axvline(-fc_thr, ls='--', lw=1.2, c='#555555')
    ax.axhline(ythr,     ls='--', lw=1.2, c='#555555')

    # shaded regions
    ax.fill_between([fc_thr, xmax],   ythr, ymax, color='#ffd6d6', alpha=0.55, zorder=0)  # top-right (bad up) RED
    ax.fill_between([-xmax, -fc_thr], ythr, ymax, color='#cfe8ff', alpha=0.55, zorder=0)  # top-left  (good up) BLUE

    # # scatter
    # plt.scatter(x[sig_other], y[sig_other], s=30, alpha=0.7, c='#7f7f7f', edgecolors='none')
    # plt.scatter(x[top_right], y[top_right], s=36, alpha=0.95, c='#1f77b4', edgecolors='none')  # blue
    # plt.scatter(x[top_left],  y[top_left],  s=36, alpha=0.95, c='#d62728', edgecolors='none')  # red

    plt.scatter(x[sig_other], y[sig_other], s=30, alpha=0.7, c='#7f7f7f', edgecolors='none')
    # 右上：bad=1 上调 ⇒ 红色
    plt.scatter(x[top_right], y[top_right], s=36, alpha=0.95, c='#d62728', edgecolors='none')
    # 左上：good=0 上调 ⇒ 蓝色
    plt.scatter(x[top_left],  y[top_left],  s=36, alpha=0.95, c='#1f77b4', edgecolors='none')


    plt.title(title, fontweight='bold')
    plt.xlabel('log2 fold change (bad vs good)', fontweight='bold')
    plt.ylabel('-log10(p-value)', fontweight='bold')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ---------------- Heatmaps etc. ----------------
def zscore_rows(mat: pd.DataFrame) -> pd.DataFrame:
    vals = mat.values
    means = np.nanmean(vals, axis=1, keepdims=True)
    stds  = np.nanstd(vals,  axis=1, keepdims=True) + 1e-8
    Z = (vals - means) / stds
    return pd.DataFrame(Z, index=mat.index, columns=mat.columns)


# === MODIFIED: heatmap routine with adjustable fontsize ======================
def _plot_heat_matrix(mat, out_png, title, cmap='jet',
                      xticks=None, yticks=None, square=True, fontsize=18):
    data = np.asarray(mat)
    H, W = data.shape
    # Larger cell size helps readability with bigger fonts
    cell_w = 0.35
    cell_h = 0.35 if square else 0.25

    fig_w = max(6, W * cell_w)
    fig_h = max(4, H * cell_h)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')

    X, Y = np.meshgrid(np.arange(W+1), np.arange(H+1))
    pcm = ax.pcolormesh(X, Y, data, cmap=cmap, shading='flat',
                        edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(pcm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize)

    if xticks is not None:
        ax.set_xticks(np.arange(0.5, W+0.5))
        ax.set_xticklabels(xticks, rotation=90, fontweight='bold', fontsize=fontsize)
    else:
        ax.set_xticks([])

    if yticks is not None:
        ax.set_yticks(np.arange(0.5, H+0.5))
        ax.set_yticklabels(yticks, fontweight='bold', fontsize=fontsize)
    else:
        ax.set_yticks([])

    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_aspect('equal' if square else 'auto')
    plt.title(title, fontweight='bold', fontsize=fontsize+1)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# === MODIFIED: larger font for sample correlation heatmap ====================
def plot_sample_corr_heatmap(sub_z: pd.DataFrame, out_png: Path,
                             title="Sample correlation (pos first, neg later)"):
    corr = np.corrcoef(sub_z.T)  # Pearson correlation
    # Replace sample IDs with P1..PN
    xticks = pnames_from(list(sub_z.columns))
    _plot_heat_matrix(corr, out_png, title, cmap='jet',
                      xticks=xticks, yticks=xticks, square=True, fontsize=18)


# === MODIFIED: abbreviate gene IDs + larger font in expression heatmap =======
def plot_expression_heatmap(sub_z: pd.DataFrame, out_png: Path,
                            title="Top DE genes (z-score) [first=bad then=good]"):
    # y-axis: ONLY Ensembl IDs (index level 0), abbreviated
    yt_full = [idx[0] for idx in sub_z.index]
    yt_short = [abbreviate_geneid(y) for y in yt_full]
    # x-axis: P1..PN
    xt = pnames_from(list(sub_z.columns))
    _plot_heat_matrix(sub_z.values, out_png, title, cmap='jet',
                      xticks=xt, yticks=yt_short, square=True, fontsize=18)


# ---------------- Single dendrogram with k=2 cut & color bars ----------------
def plot_dendrogram_two_clusters(sub: pd.DataFrame, pos_n: int, out_png: Path,
                                 title="Sample dendrogram (Ward, k=2 cut)"):
    cols = list(sub.columns)
    # P-name labels for display
    pnames = pnames_from(cols)
    true_y = np.array([1]*pos_n + [0]*(len(cols)-pos_n), dtype=int)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(sub.T)  # samples x genes
    D = pdist(X, metric='euclidean')
    Z = linkage(D, method='ward')
    Z = optimal_leaf_ordering(Z, D)

    # dendrogram with P-name labels
    plt.figure(figsize=(max(8, len(cols)*0.35), 5))
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    dendrogram(Z, labels=pnames, leaf_rotation=90, color_threshold=None)
    plt.title(title, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    # cluster cut & ARI
    leaves = leaves_list(Z)
    clabels = fcluster(Z, t=2, criterion='maxclust')  # 1 or 2
    clabels = (clabels - 1).astype(int)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(np.array(true_y), clabels[np.argsort(leaves)])
    with open(out_png.with_name(f"cluster_stats_{out_png.stem.split('_')[-1]}.txt"), "w") as f:
        f.write(f"ARI (true vs k=2 clusters): {ari:.6f}\n")

    # color bars under leaves
    ordered_pnames = np.array(pnames)[leaves]
    ordered_true  = true_y[leaves]
    ordered_pred  = clabels[leaves]
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(max(8, len(cols)*0.35), 1.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,1])
    for i, (lab, arr) in enumerate([("True (bad=1, good=0)", ordered_true),
                                    ("Cluster (k=2)", ordered_pred)]):
        axb = fig.add_subplot(gs[i, 0])
        axb.imshow(arr.reshape(1, -1), aspect='auto', cmap='bwr', vmin=0, vmax=1)
        axb.set_yticks([0]); axb.set_yticklabels([lab], fontweight='bold')
        axb.set_xticks(np.arange(len(ordered_pnames)))
        axb.set_xticklabels(ordered_pnames if i==1 else [], rotation=90, fontweight='bold')
        axb.set_facecolor('#f5f5f5')
    plt.tight_layout()
    bars_png = out_png.with_name(out_png.stem + "_bars.png")
    plt.savefig(bars_png, dpi=300)
    plt.close()


# ---------------- Consensus clustering (k configurable) ----------------
def consensus_matrix(mat: pd.DataFrame, k=2, n_iter=100, subsample=0.8, random_state=2024):
    rng = np.random.RandomState(random_state)
    n = mat.shape[1]
    C = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=float)
    Xfull = mat.T.values  # samples x genes
    for _ in range(n_iter):
        idx = rng.choice(n, size=max(2, int(round(n*subsample))), replace=False)
        X = Xfull[idx]
        km = KMeans(n_clusters=k, n_init=10, random_state=rng.randint(1e9))
        labels = km.fit_predict(X)
        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                counts[ii, jj] += 1
                if labels[i] == labels[j]:
                    C[ii, jj] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        M = np.where(counts > 0, C / counts, 0.0)
    return M


def consensus_assign_and_plot(M: np.ndarray, sample_names, out_dir: Path, tag: str, k=2):
    # Ward clustering on Euclidean of consensus
    D = pdist(M, metric='euclidean')
    Z = linkage(D, method='ward')
    Z = optimal_leaf_ordering(Z, D)
    order = leaves_list(Z)
    M_ord = M[order][:, order]

    # P-names for display
    pnames = pnames_from(sample_names)
    names_ord = np.array(pnames)[order]

    clabels = fcluster(Z, t=k, criterion='maxclust') - 1  # 0...(k-1)
    n = len(sample_names)
    half = n // 2
    true_y = np.array([1]*half + [0]*(n-half), dtype=int)  # NOTE: left as-is (not part of current request)
    inv = np.argsort(order)
    clabels_orig_order = clabels[inv]
    ari = adjusted_rand_score(true_y, clabels_orig_order)

    # Keep original sample names in CSV (analysis), only figures show P-names
    pd.DataFrame({
        "sample": sample_names,
        f"cluster_k{k}": clabels_orig_order,
        "true_label": true_y
    }).to_csv(out_dir / f"consensus_assign_k{k}_{tag}.csv", index=False)

    with open(out_dir / f"consensus_stats_k{k}_{tag}.txt", "w") as f:
        f.write(f"ARI (true vs consensus k={k}): {ari:.6f}\n")

    plt.figure(figsize=(max(6, n*0.22), max(4, n*0.22)))
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    im = ax.imshow(M_ord, vmin=0, vmax=1, interpolation='nearest', aspect='equal', cmap='jet')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names_ord, rotation=90, fontweight='bold')
    ax.set_yticklabels(names_ord, fontweight='bold')
    plt.title(f"Consensus matrix (ordered, k={k}, {tag})", fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"consensus_k{k}_{tag}.png", dpi=300)
    plt.close()


# ---------------- L1-logistic + bootstrap AUC ----------------
def logistic_auc_bootstrap(mat: pd.DataFrame, y: np.ndarray, out_dir: Path, tag="FPKM",
                           n_boot=200, random_state=2024):
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(mat.T)  # samples x genes

    clf = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, C=1.0, random_state=random_state)
    clf.fit(X, y)
    scores = clf.decision_function(X)
    fpr, tpr, _ = roc_curve(y, scores)
    auc_full = auc(fpr, tpr)

    rng = np.random.RandomState(random_state)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        Xb, yb = X[idx], y[idx]
        try:
            clf_b = LogisticRegression(penalty='l1', solver='saga', max_iter=5000, C=1.0,
                                       random_state=rng.randint(1e9))
            clf_b.fit(Xb, yb)
            scores_b = clf_b.decision_function(Xb)
            fpr_b, tpr_b, _ = roc_curve(yb, scores_b)
            aucs.append(auc(fpr_b, tpr_b))
        except Exception:
            pass

    aucs = np.array(aucs, dtype=float)
    finite_mask = np.isfinite(aucs)
    vals = aucs[finite_mask]
    # 95% CI（若无有效样本则给 NaN）
    lo = np.percentile(vals, 2.5) if vals.size > 0 else np.nan
    hi = np.percentile(vals, 97.5) if vals.size > 0 else np.nan

    # ===== ROC =====
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_full:.3f}", lw=2)
    plt.plot([0,1],[0,1], linestyle="--", color="#666666")
    plt.xlabel("FPR", fontweight='bold'); plt.ylabel("TPR", fontweight='bold'); plt.title(f"ROC ({tag})", fontweight='bold')
    plt.legend()
    plt.tight_layout(); plt.savefig(out_dir/f"roc_{tag}.png", dpi=300); plt.close()

    # ===== AUC 直方图（稳健版）=====
    plt.figure()
    if vals.size == 0:
        # 完全没有有效自举样本
        plt.text(0.5, 0.5, "No valid bootstrap AUCs", ha='center', va='center', transform=plt.gca().transAxes)
    else:
        # 若方差极小或唯一值过少，避免 bins 过多导致报错
        unique_vals = np.unique(np.round(vals, 6))
        if (vals.size < 2) or (unique_vals.size < 2) or (np.nanstd(vals) < 1e-6):
            # 退化显示：单柱直方图
            plt.hist(vals, bins=1, color="#1f77b4", alpha=0.85)
        else:
            # 根据样本量自适应 bins，上限 20
            n_bins = min(20, max(3, int(np.sqrt(vals.size))))
            plt.hist(vals, bins=n_bins, color="#1f77b4", alpha=0.85)

    mean_auc = float(np.nanmean(vals)) if vals.size > 0 else float('nan')
    plt.title(f"AUC bootstrap ({tag})  mean={mean_auc:.3f}, 95%CI[{lo:.3f},{hi:.3f}]", fontweight='bold')
    plt.tight_layout(); plt.savefig(out_dir/f"auc_bootstrap_{tag}.png", dpi=300); plt.close()

    with open(out_dir/f"auc_stats_{tag}.txt","w") as f:
        f.write(f"AUC_full={auc_full:.6f}\n")
        f.write(f"AUC_boot_mean={mean_auc:.6f}\n")
        f.write(f"AUC_boot_95CI=[{lo:.6f},{hi:.6f}]\n")
        f.write(f"n_boot={vals.size}\n")



# ---------------- Significant-gene forest / lollipop ----------------
# === MODIFIED: abbreviate IDs + larger fonts ================================
def plot_sig_gene_forest(
    res: pd.DataFrame,
    out_png: Path,
    p_thr: float = 0.05,
    fc_thr: float = 1.0,
    max_genes: Optional[int] = None,
    title: str = "Significant genes (|log2FC| with p whisker)",
    page_size: int = 68
):
    df = res.copy().reset_index()
    if "Symbol" not in df.columns:
        df["Symbol"] = df["geneid"].astype(str)

    # 按阈值筛选显著基因
    sig = df[(df["p_value"] < p_thr) & (df["log2FC"].abs() > fc_thr)].copy()
    used_as_top = False
    if len(sig) == 0:
        sig = df.sort_values("p_value").copy()
        used_as_top = True
    else:
        sig = sig.sort_values(["p_value", "log2FC"], ascending=[True, False]).copy()

    if max_genes is not None:
        sig = sig.head(max_genes)

    total = len(sig)
    n_pages = int(np.ceil(total / page_size)) if total > 0 else 0

    for pi in range(n_pages):
        part = sig.iloc[pi * page_size : (pi + 1) * page_size].copy()
        if part.empty:
            continue

        part["abs_l2fc"] = part["log2FC"].abs()
        part["neglog10p"] = -np.log10(part["p_value"].clip(lower=1e-300))

        # 计算横向误差条长度
        nl10 = part["neglog10p"].values
        if np.nanmax(nl10) > 0:
            nl10_norm = (nl10 - np.nanmin(nl10)) / (np.nanmax(nl10) - np.nanmin(nl10) + 1e-12)
        else:
            nl10_norm = np.zeros_like(nl10)
        xerr_len = 0.05 + 0.35 * (1.0 - nl10_norm)

        fig_h = max(6, 0.50 * len(part))
        plt.figure(figsize=(12, fig_h))
        ax = plt.gca()
        # 取消背景底色
        ax.set_facecolor('white')

        y_pos = np.arange(len(part))
        # 只显示基因 ID 的后六位
        labels = [r['geneid'][-6:] for _, r in part.iterrows()]
        x_vals = part["abs_l2fc"].values
        dirs = np.sign(part["log2FC"].values)
        # colors = np.where(dirs > 0, "#1f77b4", "#d62728")  # 蓝=预后差上调；红=预后好上调
        colors = np.where(dirs > 0, "#d62728", "#1f77b4")  # 红=预后差上调；蓝=预后好上调

        for yi, xv, xe, col in zip(y_pos, x_vals, xerr_len, colors):
            ax.errorbar(xv, yi, xerr=xe, fmt='o', ms=7,
                        mfc=col, mec=col, ecolor=col,
                        elinewidth=1.8, capsize=4, capthick=1.8)

        # 加粗纵轴标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontweight='bold')

        ax.set_xlabel("|log2FC| (bad vs good)", fontweight='bold')
        xmax = float(np.nanmax(x_vals)) if len(x_vals) > 0 else 1.0
        ax.set_xlim(0, xmax * 1.15 + 0.6)

        for yi, p in zip(y_pos, part["p_value"].values):
            txt = "<0.0001" if p < 1e-4 else f"{p:.4f}"
            ax.text(xmax * 1.03 + 0.45, yi, f"p={txt}", va='center', ha='left',
                    fontweight='bold', color="#333333")

        sub_title = title + ("" if not used_as_top else "  (showing top by p)")
        sub_title += f"   [Page {pi+1}/{n_pages}]"
        plt.title(sub_title, fontweight='bold')

        out_png_i = out_png if n_pages == 1 else out_png.with_name(out_png.stem + f"_part{pi+1}.png")
        out_png_i.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_png_i, dpi=300)
        plt.close()



# ---------------- GSVA helpers ----------------
def _read_gmt_simple(gmt_path: str) -> Dict[str, List[str]]:
    """简易 GMT 解析器，返回 {pathway: [genes,...]}"""
    gs = {}
    with open(gmt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                name = parts[0]
                genes = [g for g in parts[2:] if g]
                gs[name] = genes
    return gs


def load_hallmark_gene_sets(hallmark_gmt: Optional[str | Path]):
    """优先使用本地 .gmt；如果是 'auto'/None，则用 gseapy 在线获取 Hallmark（Human）。"""
    try:
        if hallmark_gmt is None or str(hallmark_gmt).lower() == "auto":
            from gseapy import get_library
            return get_library(name="Hallmark", organism="Human")  # dict
        else:
            p = Path(hallmark_gmt)
            if not p.exists():
                raise FileNotFoundError(f"{p} not found")
            return _read_gmt_simple(str(p))
    except Exception as e:
        raise RuntimeError(f"Load Hallmark gene sets failed: {e}")


def _prep_expr_for_gsva(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 all_df（index=MultiIndex[geneid, Symbol, description], columns=samples）
    变成 expr_sym（rows=Symbol, cols=samples）。同名 Symbol 取均值。
    """
    df = all_df.copy()
    df = df.reset_index()
    if "Symbol" not in df.columns:
        df["Symbol"] = df["geneid"].astype(str)
    expr_sym = df.groupby("Symbol")[all_df.columns].mean()
    expr_sym = expr_sym.dropna(how='all')
    return expr_sym


def pick_k_by_silhouette(X: np.ndarray, kmin=2, kmax=6, random_state=2024):
    best_k, best_s = kmin, -1.0
    scores = {}
    rng = np.random.RandomState(random_state)
    for k in range(kmin, kmax+1):
        labels = KMeans(n_clusters=k, n_init=20, random_state=rng.randint(1e9)).fit_predict(X)
        s = silhouette_score(X, labels)
        scores[k] = s
        if s > best_s:
            best_s, best_k = s, k
    return best_k, scores


def _plot_pathway_means_heatmap(means_df: pd.DataFrame, out_png: Path, title="GSVA pathway means (by cluster)"):
    data = means_df.values
    yticks = list(means_df.index)
    xticks = list(means_df.columns)
    _plot_heat_matrix(data, out_png, title, cmap='coolwarm', xticks=xticks, yticks=yticks, square=False, fontsize=18)


def _plot_gsva_sample_heatmap(gsva_scores: pd.DataFrame, labels: np.ndarray, out_png: Path, title="GSVA scores (samples)"):
    order = np.argsort(labels)
    S_ord = gsva_scores.iloc[:, order]
    # Map sample columns to P1..PN, then附上聚类标签
    pnames = pnames_from(list(gsva_scores.columns))
    pnames_ord = [pnames[i] for i in order]
    xticks = [f"{pnames_ord[i]}(c{labels[order[i]]})" for i in range(S_ord.shape[1])]
    _plot_heat_matrix(S_ord.values, out_png, title, cmap='coolwarm',
                      xticks=xticks, yticks=list(S_ord.index), square=False, fontsize=18)



def run_gsva_block(all_df: pd.DataFrame, pos_cols, neg_cols,
                   gmt_path: Optional[str | Path], out_dir: Path, tag: str, gsva_k: str | int = 3):
    try:
        from gseapy import gsva
    except Exception as e:
        print(f"[GSVA-{tag}] gseapy not available: {e}. Skip GSVA.")
        return

    try:
        gene_sets = load_hallmark_gene_sets(gmt_path)
    except Exception as e:
        print(f"[GSVA-{tag}] Skip GSVA: {e}")
        return

    expr_sym = _prep_expr_for_gsva(all_df)

    try:
        gsva_res = gsva(data=expr_sym,
                        gene_sets=gene_sets,
                        no_bootstrap=True,
                        sample_norm=False,
                        verbose=False,
                        min_size=10, max_size=5000, processes=1)
    except Exception as e:
        print(f"[GSVA-{tag}] GSVA failed: {e}")
        return

    gsva_scores = gsva_res.iloc[:, :]
    (out_dir / "GSVA").mkdir(parents=True, exist_ok=True)
    gsva_scores.to_csv(out_dir / "GSVA" / f"gsva_scores_{tag}.csv")

    X = gsva_scores.T.values  # samples x pathways
    if isinstance(gsva_k, str) and gsva_k.lower() == "auto":
        best_k, sc = pick_k_by_silhouette(X, kmin=2, kmax=6, random_state=2024)
        print(f"[GSVA-{tag}] auto-k by silhouette => k={best_k}, scores={sc}")
        gsva_k = int(best_k)
    else:
        gsva_k = int(gsva_k)

    km = KMeans(n_clusters=gsva_k, n_init=50, random_state=2024)
    labels = km.fit_predict(X)

    true_y = np.array([1]*len(pos_cols) + [0]*len(neg_cols), dtype=int)
    ari = adjusted_rand_score(true_y, labels) if gsva_k == 2 else np.nan

    pd.DataFrame({
        "sample": list(gsva_scores.columns),
        "gsva_cluster": labels,
        "true_label": true_y
    }).to_csv(out_dir / "GSVA" / f"gsva_assign_{tag}.csv", index=False)

    with open(out_dir / "GSVA" / f"gsva_stats_{tag}.txt", "w") as f:
        f.write(f"gsva_k={gsva_k}\n")
        f.write(f"ARI(true vs gsva_cluster)={ari:.6f}\n")

    means = []
    for c in range(gsva_k):
        means.append(gsva_scores.iloc[:, labels == c].mean(axis=1))
    means_df = pd.concat(means, axis=1)
    means_df.columns = [f"Cluster_{i}" for i in range(gsva_k)]
    _plot_pathway_means_heatmap(means_df, out_dir / "GSVA" / f"gsva_means_{tag}.png",
                                title=f"GSVA pathway means (k={gsva_k}, {tag})")

    _plot_gsva_sample_heatmap(gsva_scores, labels,
                              out_dir / "GSVA" / f"gsva_samples_{tag}.png",
                              title=f"GSVA scores (samples ordered by cluster, k={gsva_k}, {tag})")


# ---------------- 主流程块 ----------------
def run_block(pos_csv: Path, neg_csv: Path, out_dir: Path, tag: str, consensus_k: int,
              hallmark_gmt: Optional[str | Path], gsva_k: str | int):
    out_dir.mkdir(parents=True, exist_ok=True)
    pos = read_gene_table(pos_csv)
    neg = read_gene_table(neg_csv)

    if "fpkm" in tag.lower():
        pos_t = log2_fpkm(pos); neg_t = log2_fpkm(neg)
    else:
        pos_t = log2_cpm(pos);  neg_t = log2_cpm(neg)

    all_df = align_pos_neg(pos_t, neg_t)
    pos_cols = list(pos_t.columns)
    neg_cols = list(neg_t.columns)
    y = build_label_vector(pos_cols, neg_cols)

    res = welch_deg(all_df, pos_cols, neg_cols)
    res.to_csv(out_dir/f"de_results_{tag}.csv")

    plot_volcano_pretty(res, out_dir/f"volcano_{tag}.png",
                        p_thr=0.05, fc_thr=1.0, title=f"Volcano ({tag})")

    plot_sig_gene_forest(res, out_dir/f"forest_{tag}.png",
                         p_thr=0.05, fc_thr=1.0, max_genes=None,
                         title=f"Significant genes ({tag})")

    sel_idx = choose_de_genes(res, top=100)
    sub = all_df.loc[sel_idx]
    sub = sub.loc[:, pos_cols + neg_cols]  # keep pos->neg order
    sub_z = zscore_rows(sub)

    plot_expression_heatmap(sub_z, out_dir/f"heatmap_top_{tag}.png",
                            title=f"Top DE genes (z-score, {tag}) [first=bad then=good]")
    plot_sample_corr_heatmap(sub_z, out_dir/f"sample_corr_{tag}.png",
                             title=f"Sample correlation ({tag}) [first=bad then=good]")

    plot_dendrogram_two_clusters(sub, pos_n=len(pos_cols),
                                 out_png=out_dir/f"dendrogram_{tag}.png",
                                 title=f"Sample dendrogram ({tag}) [Ward + optimal leaf ordering, k=2 cut]")

    M = consensus_matrix(sub, k=consensus_k, n_iter=100, subsample=0.8, random_state=2024)
    consensus_assign_and_plot(M, list(sub.columns), out_dir, tag=tag, k=consensus_k)

    logistic_auc_bootstrap(sub, y, out_dir, tag=tag)

    run_gsva_block(all_df, pos_cols, neg_cols,
                   gmt_path=hallmark_gmt,
                   out_dir=out_dir,
                   tag=tag,
                   gsva_k=gsva_k)

    n_sig = ((res["p_value"]<0.05) & (np.abs(res["log2FC"])>1.0)).sum()
    print(f"[{tag}] genes={all_df.shape[0]}, samples={all_df.shape[1]} (pos={len(pos_cols)}, neg={len(neg_cols)})")
    print(f"[{tag}] DE genes (p<0.05 & |log2FC|>1): {n_sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpkm_pos", type=Path, required=True)
    ap.add_argument("--fpkm_neg", type=Path, required=True)
    ap.add_argument("--count_pos", type=Path, required=True)
    ap.add_argument("--count_neg", type=Path, required=True)
    ap.add_argument("--out_dir",   type=Path, required=True)
    ap.add_argument("--consensus_k", type=int, default=2, help="k for consensus clustering (default=2)")
    ap.add_argument("--hallmark_gmt", type=str, default="auto",
                    help="'auto' to fetch Hallmark via gseapy, or path to local .gmt")
    ap.add_argument("--gsva_k", type=str, default="2",
                    help="GSVA KMeans k. Integer (e.g., 2/3) or 'auto' (scan 2-6 by silhouette).")
    args = ap.parse_args()

    (args.out_dir / "FPKM").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "COUNT").mkdir(parents=True, exist_ok=True)

    run_block(args.fpkm_pos, args.fpkm_neg, args.out_dir / "FPKM",
              tag="FPKM", consensus_k=args.consensus_k,
              hallmark_gmt=args.hallmark_gmt, gsva_k=args.gsva_k)

    # run_block(args.count_pos, args.count_neg, args.out_dir / "COUNT",
    #           tag="COUNT", consensus_k=args.consensus_k,
    #           hallmark_gmt=args.hallmark_gmt, gsva_k=args.gsva_k)


if __name__ == "__main__":
    main()
