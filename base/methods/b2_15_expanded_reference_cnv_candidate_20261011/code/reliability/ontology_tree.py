"""Offline CL lookup and batch-invariant project classes from all ancestors.

【中文详解 —— Cell Ontology 离线查找与固定锚点分类】
====================================================================
在 02 的数据库映射链里承担核心角色：把 PanglaoDB / CellMarker 里的
自由文本细胞类型名，经 Cell Ontology（CL，一个细胞类型的官方层级本体）
确定地映射到项目的核心类，且全程【离线】（用 .obo 快照 + SHA256 锁定，
不联网、结果可复现）。

三个部分：
  1. CellOntologyMapper 类
     - from_files:     加载 CL 快照(校验 checksum) + 四类锚点策略 JSON，
                       同时加载原方法(Carelli)的映射函数做对照审计；
     - _resolve_input: 一个名字 → CL ID 的四级解析（显式 ID → 标准名 →
                       同义词 → 策略别名表），附带组织上下文细化；
     - resolve_many:   批量解析 + 沿 is_a 祖先链匹配"类锚点"，按
                       class_priority 决定核心类（批量顺序不影响结果）；
  2. map_marker_databases：对两个数据库的所有 (库名,细胞类型) 组合批量
     解析，产出：① PanglaoDB 的 {细胞类型: 四类 one-hot} 投票矩阵；
     ② CellMarker 的同款矩阵；③ 逐条映射审计 CSV；④ 汇总 JSON。
     关键规则：CL 里属于 Neoplastic 的类型在【投票时】记为 "Others"
     ——肿瘤判断交给 02 的 neoplastic 基因检查，而不是数据库投票。
  3. write_audit / main：审计文件落盘 / 命令行独立运行入口。

被谁调用：02_build_cluster_evidence.load_databases_compat（映射两库）。
依赖文件（相对项目根）：
    resources/cell_ontology/cl.obo          CL 本体快照
    resources/cell_ontology/snapshot.json   快照 checksum 锁
    resources/cell_ontology/four_class_policy.json  锚点/别名/细化策略
    vendor/emanuele/mapping_original.py     原方法映射函数（仅审计对照）
"""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

from .ontology import CORE_CLASSES, normalize_lookup

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = ROOT / "resources/cell_ontology"
DEFAULT_SNAPSHOT = RESOURCE_DIR / "cl.obo"
DEFAULT_POLICY = RESOURCE_DIR / "four_class_policy.json"
ORIGINAL_SOURCE = ROOT / "vendor/emanuele/mapping_original.py"


def sha256(path: Path) -> str:
    """【中文】文件 SHA256（锁定 CL 快照、策略文件、原方法源码的版本）。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original_functions(ontology) -> dict:
    """Skip only the prototype's module-level online ontology initialization.

    【中文】加载原方法(Carelli et al.)的映射函数但【跳过其联网初始化】。
    做法：AST 解析 mapping_original.py，只保留 FunctionDef 节点
    （模块级的在线 CL 下载代码被丢弃），注入运行所需的
    ontology/pd/defaultdict 后 exec。返回函数 namespace，供
    resolve_many 调用 build_cell_type_mapping / get_optimal_canonical_lineage
    做审计对照（不决定类别）。"""
    source = ast.parse(ORIGINAL_SOURCE.read_text(encoding="utf-8"))
    definitions = ast.Module(
        body=[node for node in source.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {"ms": ontology, "pd": pd, "defaultdict": defaultdict}
    exec(compile(definitions, str(ORIGINAL_SOURCE), "exec"), namespace)
    return namespace


class CellOntologyMapper:
    """Keep the author's tree for audit; classify with fixed CL ID anchors.

    【中文】CL 本体映射器。
    - "author's tree"：保留原方法的 CL 层级遍历函数（get_optimal_canonical_lineage）
      仅供审计；
    - "fixed CL ID anchors"：真正决定类别的是策略 JSON 里人工固定的
      CL ID 锚点（如 plasma cell 的 CL ID → Inflammatory），完全确定性。

    __init__ 的构建期校验：
      ① policy 必须逐字节等于打包的 four_class_policy.json（不许私改规则）；
      ② 建两张查找表：标准名→CL ID 集、同义词→CL ID 集（跳过废弃术语）；
      ③ 策略里引用的所有 CL ID 必须真实存在且不废弃；
      ④ organ_refinements 的 to_id 必须是 from_id 的祖先
        （组织细化只允许"收窄到子型"，不许乱改层级）。
    """

    def __init__(self, ontology, policy: dict, organ: str | None = None):
        if policy != json.loads(DEFAULT_POLICY.read_text(encoding="utf-8")):
            raise ValueError("Use the packaged original-method policy, not custom class rules")
        self.ontology = ontology
        self.policy = policy
        self.organ = organ
        self.original = load_original_functions(ontology)
        self.names, self.synonyms = defaultdict(set), defaultdict(set)
        for term in ontology.terms():
            if term.obsolete:
                continue
            if term.name:
                self.names[term.name.lower()].add(term.id)
            for synonym in term.synonyms:
                self.synonyms[synonym.description.lower()].add(term.id)
        ids = {i for anchors in policy["class_anchors"].values() for i in anchors}
        ids.update(row["cl_id"] for row in policy["name_aliases"])
        ids.update(rule[key] for rule in policy["organ_refinements"] for key in ("from_id", "to_id"))
        for term_id in ids:
            if term_id not in ontology or ontology[term_id].obsolete:
                raise ValueError(f"Invalid policy CL anchor: {term_id}")
        for rule in policy["organ_refinements"]:
            ancestors = {p.id for p in ontology[rule["to_id"]].superclasses(with_self=True)}
            if rule["from_id"] not in ancestors:
                raise ValueError(f"Organ refinement is not a subtype: {rule}")

    @classmethod
    def from_files(cls, snapshot=DEFAULT_SNAPSHOT, policy_path=DEFAULT_POLICY, organ=None):
        """【中文】从磁盘构建映射器（02 的 load_databases_compat 调用）。
        ① 读 snapshot.json 锁文件，比对快照文件名和 SHA256（不一致报错
           ——保证所有人用的是同一份 CL）；② pronto 库加载 .obo
        (import_depth=0 只导入本体自身)；③ 构建 mapper 并挂 provenance
        溯源字典（快照/策略/原方法源码的指纹 + 查找规则说明）。"""
        import pronto

        snapshot = Path(snapshot)
        lock = json.loads((snapshot.parent / "snapshot.json").read_text())
        if lock["file"] != snapshot.name or lock["sha256"] != sha256(snapshot):
            raise ValueError("Cell Ontology snapshot differs from its recorded checksum")
        mapper = cls(pronto.Ontology(str(snapshot), import_depth=0),
                     json.loads(Path(policy_path).read_text()), organ)
        mapper.provenance = {
            "snapshot": str(snapshot), "snapshot_sha256": lock["sha256"],
            "original_source_sha256": sha256(ORIGINAL_SOURCE),
            "policy_version": mapper.policy["version"],
            "policy_sha256": sha256(Path(policy_path)), "organ_context": organ,
            "lineage_method": "all_is_a_ancestors_fixed_CL_anchors",
            "lookup": "valid ID; canonical name; unique synonym; existing normalization; explicit qualified aliases",
            "unmapped": "excluded; no default Connective assignment",
            "input_adapter": "versioned name aliases and metadata-qualified subtype refinement",
            "normalizer_file_sha256": sha256(ROOT / "code/reliability/ontology.py"),
            "class_adapter": "Connective and Soft tissues -> Connective_tissue; Neoplastic -> Others for CDGE marker votes only",
        }
        return mapper

    @lru_cache(maxsize=None)
    def _resolve_input(self, raw: str, supplied: str) -> tuple[str, str]:
        """【中文】单个名字 → CL ID 的四级解析（lru_cache 记忆化）。
        解析顺序（先命中先停）：
          ① supplied_cl_id  调用方直接给了 CL ID（CellMarker 有 id 列），
             下划线还原成冒号后若有效 → 直接用；
          ② raw_name        名字原文（小写）查标准名表 → 同义词表；
          ③ normalized_name 归一化(ontology.normalize_lookup)后再查；
          ④ policy_alias    策略 JSON 的 name_aliases 别名表
             （按 organ 上下文过滤，"*" 通配所有组织）。
        查到 ≥2 个候选 → "ambiguous_lookup"（放弃，宁缺毋滥）；
        全部落空 → "unmatched"（该类型被排除，不默认归 Connective）。
        最后套用 organ_refinements：本组织下把 from_id 细化为 to_id
        （如乳腺 context 里某型上皮收窄为具体亚型）。
        返回: (cl_id 或 "", 解析来源说明字符串)。"""
        term_id = supplied.strip().replace("_", ":")
        if term_id in self.ontology and not self.ontology[term_id].obsolete:
            candidates, source = {term_id}, "supplied_cl_id"
        else:
            candidates, source = set(), "unmatched"
            for query, resolution in ((raw.strip().lower(), "raw_name"), (normalize_lookup(raw), "normalized_name")):
                candidates = self.names.get(query, set()) or self.synonyms.get(query, set())
                if candidates:
                    source = resolution
                    break
            if not candidates:
                aliases = [row["cl_id"] for row in self.policy["name_aliases"]
                           if row["name"] == normalize_lookup(raw) and row["organ"] in ("*", self.organ)]
                candidates, source = set(aliases), "policy_alias" if aliases else "unmatched"
        if len(candidates) != 1:
            return "", "ambiguous_lookup" if candidates else "unmatched"
        term_id = next(iter(candidates))
        for rule in self.policy["organ_refinements"]:
            if rule["organ"] == self.organ and term_id == rule["from_id"]:
                return rule["to_id"], source + "+organ_refinement"
        return term_id, source

    def query(self, name, supplied_ids="") -> str:
        """【中文】只返回解析出的 CL ID（resolve_many 的单条简化版）。"""
        raw = "" if pd.isna(name) else str(name)
        supplied = "" if pd.isna(supplied_ids) else str(supplied_ids)
        return self._resolve_input(raw, supplied)[0]

    def resolve_many(self, inputs: list[tuple[object, object]]) -> list[dict]:
        """【中文】批量解析并分类（map_marker_databases 调用一次）。
        流程：
          1. 去重后的全部查询先一次性交给原方法 build_cell_type_mapping
             （"batch-invariant" 的关键：批量入口统一，后续按条取回，
             结果与单条调用一致）；
          2. 逐条：_resolve_input 得 CL ID → 取该术语的【全部 is_a
             祖先】 → 与策略 class_anchors 求交集 → 按 class_priority
             取第一个命中的标签为核心类（全不中 = "Others"）；
          3. 同时保留原方法的目标类/谱系名做审计列；
          4. 汇总成一行字典（原名/CL ID/标准名/核心类/解析来源/祖先链/
             锚点命中/组织上下文/notes…）。
        Neoplastic 命中时 included=False → 数据库投票里记 "Others"。
        返回: 与 inputs 等长的结果字典列表。"""
        queries = list(dict.fromkeys(self.query(name, ids) for name, ids in inputs))
        original_mapping, unmapped = self.original["build_cell_type_mapping"](
            cell_list=queries, offset=0, merge_similar=False,
        )
        results = []
        for name, ids in inputs:
            raw = "" if pd.isna(name) else str(name)
            supplied = "" if pd.isna(ids) else str(ids)
            query, resolution = self._resolve_input(raw, supplied)
            term = self.ontology[query] if query else None
            ancestors = {p.id for p in term.superclasses(with_self=True)} if term else set()
            matched = {label: sorted(ancestors & set(anchors)) for label, anchors in self.policy["class_anchors"].items()}
            core = next((label for label in self.policy["class_priority"] if matched[label]), "Others")
            included = term is not None and core != "Others"
            lineage = self.original["get_optimal_canonical_lineage"](term) if term else []
            target = original_mapping.get(query, "")
            results.append({
                "raw_cell_type": "" if pd.isna(name) else str(name),
                "supplied_cl_ids": "" if pd.isna(ids) else str(ids),
                "original_query": query, "cl_id": term.id if term else "",
                "normalized_name": "" if pd.isna(name) else normalize_lookup(name),
                "query_resolution": resolution,
                "standard_name": term.name if term else "", "core_class": core,
                "original_target_class": target,
                "match_status": "mapped" if included else ("unclassified_cl_term" if term else resolution),
                "included_in_original_mapping": included,
                "ancestor_cl_ids": ";".join(sorted(ancestors)),
                "matched_anchor_ids": ";".join(matched.get(core, [])),
                "lineage_names": ";".join(lineage),
                "organ_context": self.organ or "",
                "context_query": query if "organ_refinement" in resolution or resolution == "policy_alias" else "",
                "notes": "Original target/lineage for audit only; inclusion and coarse class use the fixed B1.7 CL policy",
            })
        return results

    def resolve(self, name, supplied_ids="") -> dict:
        """【中文】单条版 resolve_many（调试/单条查询用）。"""
        return self.resolve_many([(name, supplied_ids)])[0]


def map_marker_databases(panglao, cellmarker, mapper):
    """【中文】把两个 marker 数据库映射成投票矩阵（02 的核心依赖）。
    ------------------------------------------------------------------
    输入: panglao/cellmarker 两个 DataFrame + 已构建的 CellOntologyMapper
    流程:
      1. 收集 (数据库名, 细胞类型名, CL ID, 记录数) 四元组
         ——CellMarker 带 cellontology_id 列则一并传入(走显式 ID 解析)；
      2. mapper.resolve_many 批量解析出核心类；
      3. 只保留 included_in_original_mapping=True 的行，按"同一原始名
         内所有记录核心类必须唯一"生成 one-hot 矩阵（冲突即报错）；
         Neoplastic 在投票矩阵里写 "Others"（肿瘤判断交给基因证据）；
      4. 产出 (Panglao 投票矩阵, CellMarker 投票矩阵, 审计 DataFrame,
         汇总字典)。
    02 拿前两个喂给 CDGE 的 build_gene_votes 生成基因投票表。"""
    entries = []
    for database, records, name_column, id_column in (
        ("PanglaoDB", panglao, "cell type", None),
        ("CellMarker", cellmarker, "cell_name", "cellontology_id"),
    ):
        keys = [name_column] + ([id_column] if id_column in records else [])
        selected = records.loc[records[name_column].notna()].copy()
        selected[keys] = selected[keys].fillna("")
        for key, group in selected.groupby(keys[0] if len(keys) == 1 else keys, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            entries.append((database, key[0], key[1] if len(key) > 1 else "", len(group)))
    # The original batch target is recorded, but cannot affect the fixed class.
    resolved = mapper.resolve_many([(name, ids) for _, name, ids, _ in entries])
    audit_rows = []
    for result, (database, _, _, count) in zip(resolved, entries):
        audit_rows.append({**result, "database": database, "database_records": count})
    audit = pd.DataFrame(audit_rows, columns=[
        "raw_cell_type", "supplied_cl_ids", "original_query", "normalized_name", "query_resolution", "cl_id", "standard_name",
        "core_class", "original_target_class", "match_status", "included_in_original_mapping",
        "lineage_names", "ancestor_cl_ids", "matched_anchor_ids", "organ_context", "context_query", "notes", "database", "database_records",
    ])
    maps = []
    for database, column in [("PanglaoDB", "cell_type"), ("CellMarker", "cell_name")]:
        rows = []
        kept = audit.loc[audit.database.eq(database) & audit.included_in_original_mapping.eq(True)]
        for name, group in kept.groupby("raw_cell_type", sort=True):
            vectors = []
            for core in group.core_class:
                label = "Others" if core == "Neoplastic" else core
                vectors.append({category: int(category == label) for category in CORE_CLASSES})
            unique = {tuple(vector.values()) for vector in vectors}
            if len(unique) != 1:
                raise ValueError(f"Original mapping gives conflicting classes for {database} {name!r}")
            rows.append({column: name, **vectors[0]})
        maps.append(pd.DataFrame(rows, columns=[column, *CORE_CLASSES]))
    audit["voting_class"] = audit.core_class.replace({"Neoplastic": "Others"})
    audit.loc[~audit.included_in_original_mapping.astype(bool), "voting_class"] = "excluded"
    summary = {
        **getattr(mapper, "provenance", {}), "official_labels_policy": "not_loaded",
        "name_id_records": len(audit),
        "class_counts": audit.core_class.value_counts().astype(int).to_dict(),
        "match_status_counts": audit.match_status.value_counts().astype(int).to_dict(),
        "weighted_database_records_by_class": audit.groupby("voting_class").database_records.sum().astype(int).to_dict(),
    }
    return maps[0], maps[1], audit, summary


def write_audit(out_dir, audit, summary):
    """【中文】落盘审计三件套（02/本脚本 main 调用）：
    ontology_crosswalk.csv 全部映射明细 / ontology_unresolved.csv
    未解析条目 / ontology_audit.json 汇总统计。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "ontology_crosswalk.csv", index=False)
    audit.loc[~audit.included_in_original_mapping.astype(bool)].to_csv(out_dir / "ontology_unresolved.csv", index=False)
    (out_dir / "ontology_audit.json").write_text(json.dumps(summary, indent=2))


def main():
    """【中文】独立运行入口：复用 02 的加载逻辑（importlib 动态加载该
    模块，调 load_original_voting_functions + load_databases_compat），
    传入 --tissue(必填) 过滤数据库，最终写出 crosswalk/unresolved/audit
    三个审计文件 + 两库各自的 cell_type_mapping.csv / cell_marker_mapping.csv。
    参数: --db-dir 数据库目录, --out-dir 输出目录, --tissue 组织过滤(必填)。"""
    parser = argparse.ArgumentParser(description="Original Carelli mapping with offline CL")
    parser.add_argument("--db-dir", type=Path, default=ROOT / "resources/marker_databases")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tissue", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("original_mapping_stage", ROOT / "code/02_build_cluster_evidence.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    voting = stage.load_original_voting_functions(ROOT / "vendor/cdge")
    db = stage.load_databases_compat(args.db_dir, voting, args.tissue, DEFAULT_SNAPSHOT, DEFAULT_POLICY)
    write_audit(args.out_dir, db[-1].pop("ontology_crosswalk"), db[-1]["ontology_audit"])
    db[2].to_csv(args.out_dir / "cell_type_mapping.csv", index=False)
    db[3].to_csv(args.out_dir / "cell_marker_mapping.csv", index=False)


if __name__ == "__main__":
    main()
