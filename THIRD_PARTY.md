# Third-party source and data review

This file is a release audit, not a grant of redistribution rights.

| Material in this checkout | Origin and role | Current evidence | Public release action |
| --- | --- | --- | --- |
| `vendor/CDGE/CellViT/` | CellViT model, loader, losses | Upstream [CellViT license](https://github.com/TIO-IKIM/CellViT/blob/main/LICENSE) has component-specific Apache 2.0 / Commons Clause terms and citation requirements. The local license text is retained. | Verify every vendored component, preserve notices and citation, confirm that the local snapshot and modifications comply. Weights are never bundled. |
| `vendor/CDGE/preprocessing_hest/preproce_utils/common.py` | Teacher's modified HEST patch writer | Copied from a local teacher repository whose configured GitHub remote was not publicly accessible during this audit. No license file for that modification was identified in the handoff. | Obtain the author's permission/license or replace this dependency with a separately licensed implementation before public release. |
| `base/methods/**/vendor/cdge/` and `vendor/emanuele/` | Teacher and collaborator code used by the molecular workflow | No redistribution grant was found in the handoff. | Obtain permission/license or make it an explicitly installed external dependency. Preserve scientific behavior in either case. |
| `base/methods/**/resources/marker_databases/` | Marker database snapshots and project mappings | The copied files do not carry a redistribution notice. The [PanglaoDB site](https://panglaodb.se/markers.html) and CellMarker publication describe data access, but access alone does not establish this checkout's redistribution rights. | Trace the exact downloaded files, versions, notices, and rights with the maintainers before public release. |
| `base/methods/**/resources/cell_ontology/` | Cell Ontology snapshot and mappings | [Cell Ontology project](https://github.com/obophenotype/cell-ontology) publishes ontology releases; the copied snapshot's origin/version and redistribution notice have not been verified. | Record exact source revision and applicable notice; preserve attribution. |
| HEST Xenium, PanNuke, H&E, model weights | User-supplied research inputs | None are intentionally included in this repository. | Keep out of Git; follow each source's access and usage rules separately. |

The teacher's repository remote in the checked server was
`https://github.com/naivete5656/CDGE.git` at commit
`d97e0e2ab658b74903b42c7d95c16c29eec6f539`. It returned 404 to an
unauthenticated lookup during this audit. This does **not** prove the code is
unlicensed; it means public redistribution was not confirmed.

The preserved molecular bundle includes original source snapshots, with hashes
in `base/BUNDLE_MANIFEST.json` and `provenance/SOURCE_MAP.json`. Seven copied
Python files received **path-only** changes: six CNV scripts now require
`--adata`, and the historical direct-run classifier no longer points to the
old server. The active wrapper already passed explicit paths. Original and
packaged hashes remain separately recorded; no scientific formula changed.

Do not add a blanket repository license that purports to relicense these
materials. The research authors should decide the license for their own code
after third-party rights are settled.
