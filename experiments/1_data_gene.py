#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, Tuple, List

import polars as pl


CORE_COLS = ["geneid", "Symbol", "description"]


def read_slide_ids(slide_pred_csv: Path) -> Tuple[List[str], List[str]]:
    """从 slide_pred.csv 读取 y_true=y_pred=1 和 y_true=y_pred=0 的 slide_id 列表。"""
    df = pl.read_csv(slide_pred_csv)
    for c in ["slide_id", "y_true", "y_pred"]:
        if c not in df.columns:
            raise RuntimeError(f"{slide_pred_csv} 缺少列: {c}")
    df = df.with_columns(
        pl.col("slide_id").cast(pl.Utf8),
        pl.col("y_true").cast(pl.Int64),
        pl.col("y_pred").cast(pl.Int64),
    )

    pos_ids = (
        df.filter((pl.col("y_true") == 1) & (pl.col("y_pred") == 1))
          .select("slide_id")["slide_id"]
          .to_list()
    )
    neg_ids = (
        df.filter((pl.col("y_true") == 0) & (pl.col("y_pred") == 0))
          .select("slide_id")["slide_id"]
          .to_list()
    )
    return pos_ids, neg_ids


def make_table(
    gene_df: pl.DataFrame,
    slide_ids: Iterable[str],
) -> Tuple[pl.DataFrame, List[str]]:
    """
    从 gene_df 里抽取 CORE_COLS + 每个 slide_id 对应的病理号列。
    - 匹配列时用 base_id = slide_id.split('-')[0]
    - 导出时列名重命名为 slide_id
    - 若同一 base_id 对应多个 slide_id，会复制该列多次并分别命名
    返回：结果 DataFrame，以及未匹配到基因表的 slide_id 列表
    """
    cols = set(gene_df.columns)
    missing: List[str] = []

    # 先检查 CORE_COLS
    for c in CORE_COLS:
        if c not in cols:
            raise RuntimeError(f"基因表缺少必要列：{c}")

    exprs = [pl.col(c) for c in CORE_COLS]

    for sid in slide_ids:
        base = sid.split("-")[0]
        if base in cols:
            # 复制该列，并重命名为 slide_id
            exprs.append(pl.col(base).alias(sid))
        else:
            missing.append(sid)

    if len(exprs) == len(CORE_COLS):
        # 没有任何匹配到的 slide 列，仍返回只有 CORE_COLS 的空壳（行数同原表）
        return gene_df.select(exprs), missing

    out = gene_df.select(exprs)
    return out, missing


def main(slide_pred: Path, count_csv: Path, fpkm_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取预测正确的 slide_id（正负两组）
    pos_ids, neg_ids = read_slide_ids(slide_pred)

    # 读基因表
    count_df = pl.read_csv(count_csv)
    fpkm_df  = pl.read_csv(fpkm_csv)

    # 逐表生成四个输出
    # 1) label_1_* （正样本）
    pos_count, miss_pos_count = make_table(count_df, pos_ids)
    pos_fpkm,  miss_pos_fpkm  = make_table(fpkm_df,  pos_ids)

    # 2) label_0_* （负样本）
    neg_count, miss_neg_count = make_table(count_df, neg_ids)
    neg_fpkm,  miss_neg_fpkm  = make_table(fpkm_df,  neg_ids)

    # 写出
    p1 = out_dir / "label_1_count.csv"
    p2 = out_dir / "label_1_fpkm.csv"
    p3 = out_dir / "label_0_count.csv"
    p4 = out_dir / "label_0_fpkm.csv"

    pos_count.write_csv(p1)
    pos_fpkm.write_csv(p2)
    neg_count.write_csv(p3)
    neg_fpkm.write_csv(p4)

    # 汇报
    print(f"[OK] 写出: {p1}  列数={pos_count.width}  行数={pos_count.height}")
    if miss_pos_count:
        print(f"    ⛔ 未在 Count 表中找到（按病理号列）: {len(miss_pos_count)} 个，如: {miss_pos_count[:5]}")
    print(f"[OK] 写出: {p2}  列数={pos_fpkm.width}  行数={pos_fpkm.height}")
    if miss_pos_fpkm:
        print(f"    ⛔ 未在 FPKM 表中找到（按病理号列）: {len(miss_pos_fpkm)} 个，如: {miss_pos_fpkm[:5]}")
    print(f"[OK] 写出: {p3}  列数={neg_count.width}  行数={neg_count.height}")
    if miss_neg_count:
        print(f"    ⛔ 未在 Count 表中找到（按病理号列）: {len(miss_neg_count)} 个，如: {miss_neg_count[:5]}")
    print(f"[OK] 写出: {p4}  列数={neg_fpkm.width}  行数={neg_fpkm.height}")
    if miss_neg_fpkm:
        print(f"    ⛔ 未在 FPKM 表中找到（按病理号列）: {len(miss_neg_fpkm)} 个，如: {miss_neg_fpkm[:5]}")

    # 额外小提示：展示一下成功匹配的 slide 列数量（去掉核心三列）
    def matched_cols(df: pl.DataFrame) -> int:
        return max(df.width - len(CORE_COLS), 0)

    print("\n=== 匹配统计（按列）===")
    print(f"label_1_count: {matched_cols(pos_count)}")
    print(f"label_1_fpkm : {matched_cols(pos_fpkm)}")
    print(f"label_0_count: {matched_cols(neg_count)}")
    print(f"label_0_fpkm : {matched_cols(neg_fpkm)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide_pred", type=Path, required=True,
                    help="预测结果 CSV（含 slide_id, y_true, y_pred）")
    ap.add_argument("--count_csv",  type=Path, required=True,
                    help="All_TNBC_Count_new.csv")
    ap.add_argument("--fpkm_csv",   type=Path, required=True,
                    help="All_TNBC_FPKM_new.csv")
    ap.add_argument("--out_dir",    type=Path, required=True,
                    help="输出目录（将写出四个 label_*_*.csv）")
    args = ap.parse_args()
    main(**vars(args))
