#!/usr/bin/env python3
"""Fetch standard GRCh38 coordinates for one or more Xenium panels.

This is a data-compatibility helper for the unchanged InSituCNV workflow.
It reads only panel gene names and queries Ensembl's public REST endpoint;
no cell labels, cluster labels, or evaluation annotations are opened.
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
    """Split a failed batch so control features do not block real genes."""
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
