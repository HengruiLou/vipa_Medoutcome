#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import random
from typing import List, Tuple

import polars as pl

CORE = ["geneid", "Symbol", "description"]


def read_priority_ids(label_csv: Path) -> List[str]:
    """
    读取 label_?.csv（需含 slide_id, prob），按 prob 降序返回 slide_id 列表。
    如果未排序，函数内会按 prob 排序。
    """
    df = pl.read_csv(label_csv)
    if "slide_id" not in df.columns or "prob" not in df.columns:
        raise RuntimeError(f"{label_csv} 必须包含列 slide_id 与 prob")
    df = df.select(["slide_id", "prob"]).sort("prob", descending=True)
    return df["slide_id"].to_list()


def pick_topk_columns(
    gene_df: pl.DataFrame,
    priority_ids: List[str],
    k: int,
    seed: int,
) -> Tuple[pl.DataFrame, int, int, List[str], List[str]]:
    """
    在 gene_df 中，根据 priority_ids（按 prob 降序）挑选 k 列：
    - 命中的优先列（出现在 gene_df 列名里）优先加入；
    - 若不够 k，从剩余列随机补齐；
    返回：(输出 DataFrame, 命中数量, 随机补充数量, 用到的列名列表, 随机补充的列名列表)
    """
    cols = gene_df.columns
    for c in CORE:
        if c not in cols:
            raise RuntimeError("基因表缺少必要列: " + c)

    # 可用的“病理号-编号”列（除去 CORE）
    pool = [c for c in cols if c not in CORE]

    # 1) 命中的优先列（保持顺序 & 去重）
    chosen: List[str] = []
    for sid in priority_ids:
        if sid in pool and sid not in chosen:
            chosen.append(sid)
        if len(chosen) >= k:
            break

    hit = len(chosen)

    # 2) 不足则随机补齐
    if len(chosen) < k:
        random.seed(seed)
        remain = [c for c in pool if c not in chosen]
        need = k - len(chosen)
        if len(remain) > 0:
            add = random.sample(remain, k=min(need, len(remain)))
            chosen.extend(add)
        else:
            add = []
    else:
        add = []

    rnd = len(add)

    # 3) 仅选择 CORE + 选中的列
    out_cols = CORE + chosen
    out_df = gene_df.select(out_cols)

    return out_df, hit, rnd, chosen, add


def process_one(
    label_csv: Path,
    gene_csv: Path,
    out_path: Path,
    k: int,
    seed: int,
    tag: str,
):
    """对单个（label组 × 基因表）执行选择并写出。"""
    priority = read_priority_ids(label_csv)
    gene_df = pl.read_csv(gene_csv)

    out_df, hit, rnd, chosen, add = pick_topk_columns(gene_df, priority, k, seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_csv(out_path)

    print(f"[OK] {tag}: 写出 {out_path.name}  行数={out_df.height}  列数={out_df.width}")
    print(f"     ▶ 优先命中 {hit} 个；随机补充 {rnd} 个；目标={k} 个")
    if hit < k:
        # 简短展示前几个未命中/随机列
        missed_cnt = len([sid for sid in priority if sid not in gene_df.columns])
        print(f"     ⛔ 优先列表中有 {missed_cnt} 个 slide_id 不在该基因表列里（按列名匹配）。")
        print(f"     ➕ 随机补充示例: {add[:5] if add else []}")
    print(f"     ✓ 最终列（除去CORE）数量={max(out_df.width - len(CORE), 0)}\n")


def main(
    label0: Path, label1: Path,
    count0: Path, count1: Path,
    fpkm0: Path,  fpkm1: Path,
    out_dir: Path,
    k: int, seed: int
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 四个输出
    process_one(
        label_csv=label0,
        gene_csv=count0,
        out_path=out_dir/"label_0_count_top20.csv",
        k=k, seed=seed, tag="NEG-COUNT"
    )
    process_one(
        label_csv=label1,
        gene_csv=count1,
        out_path=out_dir/"label_1_count_top20.csv",
        k=k, seed=seed, tag="POS-COUNT"
    )
    process_one(
        label_csv=label0,
        gene_csv=fpkm0,
        out_path=out_dir/"label_0_fpkm_top20.csv",
        k=k, seed=seed, tag="NEG-FPKM"
    )
    process_one(
        label_csv=label1,
        gene_csv=fpkm1,
        out_path=out_dir/"label_1_fpkm_top20.csv",
        k=k, seed=seed, tag="POS-FPKM"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label0", type=Path, required=True, help="label_0.csv（含 slide_id, prob）")
    ap.add_argument("--label1", type=Path, required=True, help="label_1.csv（含 slide_id, prob）")
    ap.add_argument("--count0", type=Path, required=True, help="label_0_count.csv（列名为病理号-编号 + CORE）")
    ap.add_argument("--count1", type=Path, required=True, help="label_1_count.csv")
    ap.add_argument("--fpkm0",  type=Path, required=True, help="label_0_fpkm.csv")
    ap.add_argument("--fpkm1",  type=Path, required=True, help="label_1_fpkm.csv")
    ap.add_argument("--out_dir", type=Path, required=True, help="输出目录，生成 *_top20.csv")
    ap.add_argument("--k", type=int, default=20, help="每个输出文件保留的列数（不含 CORE）")
    ap.add_argument("--seed", type=int, default=2024, help="随机补充的随机种子")
    args = ap.parse_args()
    main(**vars(args))
