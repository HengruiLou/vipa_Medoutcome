#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl


def load_train_ids(train_list_path: Path) -> set[str]:
    """读取 train_list.txt，剥去 .npy 得到 slide_id 集合"""
    lines = Path(train_list_path).read_text(encoding="utf-8").splitlines()
    ids = set()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        # 取文件名去掉扩展名   e.g. "201479733-3.npy" -> "201479733-3"
        sid = Path(ln).stem
        ids.add(sid)
    return ids


def main(slide_pred: Path, train_list: Path, out_dir: Path | None):
    slide_pred = Path(slide_pred)
    if out_dir is None:
        out_dir = slide_pred.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取训练集 slide_id
    train_ids = load_train_ids(train_list)

    # 读取预测结果
    df = pl.read_csv(slide_pred)

    # 规范列名与类型
    required_cols = {"slide_id", "y_true", "y_pred", "prob"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise RuntimeError(f"slide_pred.csv 缺少列: {missing}")

    # slide_id 去掉可能的后缀/路径噪声
    df = df.with_columns(
        pl.col("slide_id").map_elements(lambda s: Path(str(s)).stem, return_dtype=pl.Utf8),
        pl.col("y_true").cast(pl.Int64),
        pl.col("y_pred").cast(pl.Int64),
        pl.col("prob").cast(pl.Float64),
    )

    # 仅保留：预测正确 + 不在训练集
    df_keep = df.filter(
        (pl.col("y_true") == pl.col("y_pred"))
        & (~pl.col("slide_id").is_in(list(train_ids)))
    )

    # 分标签、排序、写出
    for lab in (0, 1):
        sub = (
            df_keep
            .filter(pl.col("y_true") == lab)   # y_true==y_pred，此处等价于用 y_pred
            .sort("prob", descending=True)
        )
        out_csv = out_dir / f"label_{lab}.csv"
        sub.write_csv(out_csv)
        print(f"[OK] 写出 {out_csv}  行数={sub.height}")

    # 额外统计信息
    total = df.shape[0]
    correct = df.filter(pl.col("y_true") == pl.col("y_pred")).shape[0]
    kept = df_keep.shape[0]
    print(f"\n总样本: {total} | 预测正确: {correct} | 预测正确且非训练集: {kept}")
    print(f"label_0: {(df_keep['y_true']==0).sum()}  |  label_1: {(df_keep['y_true']==1).sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="筛选预测正确且非训练集的样本，按标签分别保存")
    ap.add_argument("--slide_pred", type=Path, required=True, help="slide_pred.csv 路径")
    ap.add_argument("--train_list", type=Path, required=True, help="train_list.txt 路径（含 .npy 文件名）")
    ap.add_argument("--out_dir", type=Path, default=None, help="输出目录（默认与 slide_pred.csv 同目录）")
    args = ap.parse_args()
    main(**vars(args))
