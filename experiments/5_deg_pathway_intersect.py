#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path


# ---------- 工具函数 ----------

def shorten_enstid(gid: str) -> str:
    """
    将基因ID转换成你 txt 文件里的“后六位”形式：
    逻辑：
        1) 去掉版本号（以 '.' 分割）
        2) 若以 'ENSG00000' 开头，则去掉该前缀
        3) 取最后 6 个字符
    例：
        'ENSG00000000003'  -> '000003'
        'ENSG00000123456.7' -> '123456'
    """
    s = str(gid).split(".", 1)[0]
    if s.startswith("ENSG00000"):
        core = s[len("ENSG00000"):]
    else:
        core = s
    if len(core) > 6:
        core = core[-6:]
    return core


def read_gmt(gmt_path: Path):
    """读取 .gmt 文件，返回 {pathway_name: [gene_symbol,...]}"""
    gene_sets = {}
    with open(gmt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            gs_name = parts[0]
            genes = parts[2:]
            gene_sets[gs_name] = genes
    return gene_sets


def main():
    # === 路径（如需调整你可以手动改这里） ===
    base_dir = Path("/medical-data/lhr/娄恒瑞预后/基因")
    deg_csv = base_dir / "analysis_out/FPKM/de_results_FPKM.csv"
    id_txt = base_dir / "136显著基因.txt"
    gmt_path = base_dir / "h.all.v2025.1.Hs.symbols.gmt"
    out_dir = base_dir / "analysis_out/FPKM"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_136 = out_dir / "deg_136_genes.csv"
    out_intersect = out_dir / "deg_pathway_intersect.csv"
    out_sig_genes = out_dir / "deg_sig_genes.csv"

    # ---- 1. 读取 136 个显著基因 ID（后六位形式） ----
    if not id_txt.exists():
        raise FileNotFoundError(f"显著基因 ID 文件不存在: {id_txt}")

    with open(id_txt, "r", encoding="utf-8") as f:
        id_list = [ln.strip() for ln in f if ln.strip()]
    id_set = set(id_list)
    print(f"[INFO] 从 {id_txt} 读取到 {len(id_set)} 个显著基因短ID（预期=136）")

    # ---- 2. 读取 de_results_FPKM.csv，并匹配 136 个基因 ----
    if not deg_csv.exists():
        raise FileNotFoundError(f"de_results_FPKM.csv 不存在: {deg_csv}")

    df = pd.read_csv(deg_csv)
    if "geneid" not in df.columns or "Symbol" not in df.columns:
        raise RuntimeError("de_results_FPKM.csv 中必须包含 'geneid' 和 'Symbol' 列")

    # 生成一列 'short_id' 与 txt 中的 ID 对应
    df["short_id"] = df["geneid"].apply(shorten_enstid)

    sig = df[df["short_id"].isin(id_set)].copy()
    print(f"[INFO] 在 de_results_FPKM.csv 中匹配到 {len(sig)} 个基因")

    if len(sig) != len(id_set):
        print("[WARN] 匹配到的基因数量与 txt 中 ID 数量不一致，"
              "可能有某些 ID 在 de_results_FPKM.csv 中找不到。")

    # 去掉辅助列 short_id，只保留原始信息
    sig = sig.drop(columns=["short_id"])

    # 保存 136 个显著基因详细信息
    sig.to_csv(out_136, index=False, encoding="utf-8-sig")
    print(f"[OK] 136 个显著基因信息已保存到: {out_136}")

    # 构建显著基因 Symbol 集合（用于通路交集）
    sig["Symbol"] = sig["Symbol"].astype(str)
    sig_genes = set(sig["Symbol"])
    print(f"[INFO] 用于通路分析的显著基因 Symbol 数量: {len(sig_genes)}")

    # ---- 3. 读取 Hallmark.gmt ----
    if not gmt_path.exists():
        raise FileNotFoundError(f"Hallmark GMT 文件不存在: {gmt_path}")

    gene_sets = read_gmt(gmt_path)
    print(f"[INFO] Hallmark 通路数：{len(gene_sets)}")

    # ---- 4. 通路维度：统计每条通路和显著基因的交集 ----
    rows = []
    # 为基因维度统计预先准备 gene -> pathways 映射
    gene2pathways = {g: set() for g in sig_genes}

    for pathway, genes in gene_sets.items():
        gs = set(genes)
        inter = gs & sig_genes  # 交集
        for g in inter:
            gene2pathways[g].add(pathway)

        rows.append({
            "pathway": pathway,
            "n_overlap": len(inter),
            "overlap_genes": ";".join(sorted(inter)),
        })

    intersect_df = pd.DataFrame(rows).sort_values("n_overlap", ascending=False)
    intersect_df.to_csv(out_intersect, index=False, encoding="utf-8-sig")
    print(f"[OK] 通路×显著基因交集结果已保存: {out_intersect}")

    # ---- 5. 基因维度：为 136 个显著基因附上它所在的 Hallmark 通路 ----
    n_list = []
    pw_list = []
    for _, row in sig.iterrows():
        sym = str(row["Symbol"])
        pw = sorted(gene2pathways.get(sym, []))
        n_list.append(len(pw))
        pw_list.append(";".join(pw))

    sig_with_pw = sig.copy()
    sig_with_pw["n_hallmark_overlap"] = n_list          # 该基因落在多少条 Hallmark 通路中
    sig_with_pw["overlap_pathways"] = pw_list           # 具体哪些通路

    sig_with_pw.to_csv(out_sig_genes, index=False, encoding="utf-8-sig")
    print(f"[OK] 显著基因 + 通路信息已保存: {out_sig_genes}")


if __name__ == "__main__":
    main()
