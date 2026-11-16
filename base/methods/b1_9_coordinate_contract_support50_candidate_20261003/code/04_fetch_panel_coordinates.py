#!/usr/bin/env python3
"""Fetch standard GRCh38 coordinates for one or more Xenium panels.

This is a data-compatibility helper for the unchanged InSituCNV workflow.
It reads only panel gene names and queries Ensembl's public REST endpoint;
no cell labels, cluster labels, or evaluation annotations are opened.

【中文详解 —— 流水线第 4 步：给 panel 基因查 GRCh38 基因组坐标】
====================================================================
流水线位置：01→02→03 之后、05(InSituCNV) 之前。05 的 CNV 推断必须知道
每个基因在哪条染色体的哪个区间（按染色体位置开窗），本脚本就是负责
把 Xenium panel 的基因名（约 280 个）变成 基因→(chr,start,end) 坐标表。

做什么：
    从 h5ad(--adata) 或纯文本基因列表(--genes-file)收集 panel 基因名，
    过滤掉探针控制项（BLANK/NEGCONTROL/...），优先复用已有坐标表
    (--base-coordinates，可多次传入)，缺失的基因批量(默认 50 个/批)查询
    Ensembl REST API（https://rest.ensembl.org，POST lookup/symbol），
    只保留 GRCh38 结果，统一染色体命名（如 "1"→"chr1"），按染色体位置
    排序输出 TSV。

输入 → 输出：
    panel 基因名 + 可选旧坐标表 → coordinates TSV（gene/chromosome/start/
    end/ensembl_id 五列），供 05 的 build_gene_reference 使用。

与其他模块的连接：
    调用 reliability/data_contracts.normalize_genomic_coordinates
    ——坐标进表前后都做规范化校验（染色体名、区间合法性、基因名唯一）。
    本脚本不读任何细胞标签或注释，与评估完全隔离。

健壮性设计：
    lookup_resilient 对每批最多重试 4 次（指数退避 sleep 2/4/6s），
    仍失败就把批次二分递归——保证一个坏基因（如控制探针残留）不会
    拖垮整批真实基因的查询。
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import socket
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from reliability.data_contracts import normalize_genomic_coordinates



def parse_args() -> argparse.Namespace:
    """【中文】命令行参数。
    --adata             一个或多个 h5ad（取其 var_names 当基因列表）
                        ——与 --genes-file 二选一（互斥必填组）
    --genes-file        纯文本基因列表（每行一个基因符号）
    --base-coordinates  可重复传入；已有的坐标 TSV，先复用、缺的才联网查
    --out               输出 TSV 路径
    --batch-size        每批查询的基因数（Ensembl 上限 50），默认 50
    """
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--adata", type=Path, nargs="+")
    source.add_argument("--genes-file", type=Path)
    parser.add_argument(
        "--base-coordinates",
        type=Path,
        action="append",
        default=[],
        help="Existing TSV coordinate table to reuse before querying missing genes.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


def lookup_batch(symbols: list[str]) -> dict[str, object]:
    """【中文】单批查询：POST {"symbols":[...]} 到 Ensembl REST，60 秒超时。
    返回 {基因符号: {seq_region_name,start,end,id,assembly_name,...}}。
    被 lookup_resilient 调用。"""
    payload = json.dumps({"symbols": symbols}).encode("utf-8")
    request = Request(
        "https://rest.ensembl.org/lookup/symbol/homo_sapiens?expand=0",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:  # nosec B310: official public endpoint
        return json.loads(response.read().decode("utf-8"))


def lookup_resilient(symbols: list[str]) -> dict[str, object]:
    """Split a failed batch so control features do not block real genes.

    【中文】带重试与二分容错的批量查询（main 对每个缺失批次调用它）。
    策略：整批重试最多 4 次（间隔 2/4/6 秒递增）；仍失败且批内 >1 个基因
    就对半拆成两半分别递归——最终单个"查不到/总报错"的基因只会让自己
    返回空字典，不影响同批其他基因。全部查完返回合并结果字典。
    """
    for attempt in range(4):
        try:
            return lookup_batch(symbols)
        except (HTTPError, URLError, TimeoutError, socket.timeout, ssl.SSLError, http.client.RemoteDisconnected):
            if attempt < 3:
                time.sleep(2 * (attempt + 1))
    if len(symbols) == 1:
        return {}
    middle = len(symbols) // 2
    return lookup_resilient(symbols[:middle]) | lookup_resilient(symbols[middle:])


def read_panel_genes(adata_paths: list[Path] | None, genes_file: Path | None) -> list[str]:
    """【中文】收集 panel 基因名并保持出现顺序、去重（dict 键序实现）。
    adata 路径用 backed="r" 只读元数据（不加载计数矩阵，省内存），
    基因名统一大写。被 main 调用。"""
    genes: dict[str, None] = {}
    if adata_paths:
        import anndata as ad

        for path in adata_paths:
            for gene in ad.read_h5ad(path, backed="r").var_names.astype(str).str.upper().unique():
                genes.setdefault(gene, None)
    else:
        for line in genes_file.read_text().splitlines():
            gene = line.strip().upper()
            if gene:
                genes.setdefault(gene, None)
    return list(genes)


def read_base_coordinates(paths: list[Path], panel_genes: set[str]) -> list[dict[str, object]]:
    """【中文】读入可复用的旧坐标表，只保留 panel 需要的基因。
    要求 TSV 含 gene/chromosome/start/end 四列（缺列直接报错）；
    同一基因以最先出现的表为准（后续表的同名基因被忽略，基因名先大写）。
    读入后过 normalize_genomic_coordinates 规范化（染色体名统一 chr 前缀、
    区间校验）。返回记录列表（供 main 与新查询结果合并）。"""
    rows: dict[str, dict[str, object]] = {}
    required = {"gene", "chromosome", "start", "end"}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Coordinate table {path} misses columns: {sorted(missing)}")
            for row in reader:
                gene = str(row["gene"]).strip().upper()
                if gene not in panel_genes or gene in rows:
                    continue
                rows[gene] = {
                    "gene": gene,
                    "chromosome": row["chromosome"],
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "ensembl_id": row.get("ensembl_id", ""),
                }
    if not rows:
        return []
    table = normalize_genomic_coordinates(pd.DataFrame(rows.values()).rename(columns={"gene": "gene_name"}))
    return table.rename(columns={"gene_name": "gene"}).drop(columns="chromosome_original").to_dict("records")


def main() -> None:
    """【中文】主流程：
      1. 收集 panel 基因，剔除 4 类探针控制项（BLANK / NEGCONTROL /
         ANTISENSE / UNASSIGNED——它们不是真基因，查不到坐标是正常的）；
      2. 复用旧坐标表，得到仍缺坐标的基因名单；
      3. 分批联网查询（批间 sleep 0.5 秒限流），只接受
         assembly_name == "GRCh38" 的记录，染色体名统一加 chr 前缀；
      4. 合并去重 → 规范化校验（一个坐标都没有则 RuntimeError）→
         按 (染色体, start) 排序 → 写五列 TSV → 打印统计 JSON。
    """
    args = parse_args()
    panel_genes = read_panel_genes(args.adata, args.genes_file)
    genes = [
        gene for gene in panel_genes
        if not any(token in gene for token in ("BLANK", "NEGCONTROL", "ANTISENSE", "UNASSIGNED"))
    ]
    rows = read_base_coordinates(args.base_coordinates, set(genes))
    reused = {str(row["gene"]) for row in rows}
    missing_genes = [gene for gene in genes if gene not in reused]
    for start in range(0, len(missing_genes), args.batch_size):
        batch = missing_genes[start : start + args.batch_size]
        result = lookup_resilient(batch)
        for symbol in batch:
            record = result.get(symbol)
            if not record or record.get("assembly_name") != "GRCh38":
                continue
            chromosome = str(record["seq_region_name"])
            if not chromosome.startswith("chr"):
                chromosome = f"chr{chromosome}"
            rows.append(
                {
                    "gene": symbol,
                    "chromosome": chromosome,
                    "start": int(record["start"]),
                    "end": int(record["end"]),
                    "ensembl_id": record.get("id", ""),
                }
            )
        print(
            f"looked up {min(start + len(batch), len(missing_genes))}/{len(missing_genes)} missing panel genes",
            flush=True,
        )
        time.sleep(.5)

    deduplicated = {str(row["gene"]): row for row in rows}
    if not deduplicated:
        raise RuntimeError("No GRCh38 panel gene coordinates could be resolved")
    normalized = normalize_genomic_coordinates(pd.DataFrame(deduplicated.values()).rename(columns={"gene": "gene_name"}))
    records = normalized.rename(columns={"gene_name": "gene"}).drop(columns="chromosome_original").to_dict("records")
    table = sorted(records, key=lambda row: (str(row["chromosome"]), int(row["start"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["gene", "chromosome", "start", "end", "ensembl_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({
        "input_panels": [str(path) for path in args.adata] if args.adata else [],
        "candidate_gene_symbols": len(genes),
        "coordinates_reused": len(reused),
        "symbols_queried": len(missing_genes),
        "grch38_coordinates_found": len(table),
        "official_labels_policy": "not_loaded",
    }, indent=2))


if __name__ == "__main__":
    main()
