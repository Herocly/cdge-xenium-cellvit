"""Offline CL lookup and batch-invariant project classes from all ancestors."""

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original_functions(ontology) -> dict:
    """Skip only the prototype's module-level online ontology initialization."""
    source = ast.parse(ORIGINAL_SOURCE.read_text(encoding="utf-8"))
    definitions = ast.Module(
        body=[node for node in source.body if isinstance(node, ast.FunctionDef)],
        type_ignores=[],
    )
    namespace = {"ms": ontology, "pd": pd, "defaultdict": defaultdict}
    exec(compile(definitions, str(ORIGINAL_SOURCE), "exec"), namespace)
    return namespace


class CellOntologyMapper:
    """Keep the author's tree for audit; classify with fixed CL ID anchors."""

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
        raw = "" if pd.isna(name) else str(name)
        supplied = "" if pd.isna(supplied_ids) else str(supplied_ids)
        return self._resolve_input(raw, supplied)[0]

    def resolve_many(self, inputs: list[tuple[object, object]]) -> list[dict]:
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
        return self.resolve_many([(name, supplied_ids)])[0]


def map_marker_databases(panglao, cellmarker, mapper):
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
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "ontology_crosswalk.csv", index=False)
    audit.loc[~audit.included_in_original_mapping.astype(bool)].to_csv(out_dir / "ontology_unresolved.csv", index=False)
    (out_dir / "ontology_audit.json").write_text(json.dumps(summary, indent=2))


def main():
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
