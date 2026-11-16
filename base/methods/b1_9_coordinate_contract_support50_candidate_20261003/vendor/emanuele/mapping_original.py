from collections import defaultdict
import pandas as pd

# define the mapping space from ontology
import pronto
ms = pronto.Ontology("http://purl.obolibrary.org/obo/cl.owl")

def define_macro_classes(auto=True, manually_defined_macro_classes=None):
    if auto:
        return ['neoplastic', 'epithelial', 'leukocyte', 'connective', 'endothelial']
    else:
        return manually_defined_macro_classes

def return_term_ontology(term_input, ontology = None):
    # 1. check obj Term
    if not isinstance(term_input, str):
        return term_input

    if ontology is None:
        ontology = ms

    query_clean = term_input.strip()
    query_lower = query_clean.lower()

    if not query_clean:
        return None

    # 2. exact id ("CL:0000890" or "CL_0000890")
    formatted_id = query_clean.replace("_", ":")
    if formatted_id in ontology:
        return ontology[formatted_id]

    # 3. name (case-insensitive)
    for t in ontology.terms():
        if t.name and t.name.lower() == query_lower:
            return t

    # 4. synonims (case-insensitive)
    for t in ontology.terms():
        if t.synonyms:
            for syn in t.synonyms:
                if syn.description and syn.description.lower() == query_lower:
                    return t

    # 5. error if None
    print(f"Term '{term_input}' not found.")
    return None

def get_most_specific_parent(parents):
    """Selects the parent node whose subtree contains the highest number of subclasses.

    This prioritizes deep biological lineage branches over shallow functional
    groupings.
    """
    best_parent = None
    max_descendants = -1

    for p in parents:
        # Count all recursive subclasses (descendants) in the ontology for this parent
        # with_self=False counts only the children/descendants below it
        num_descendants = len(list(p.subclasses(with_self=False)))

        if num_descendants > max_descendants:
            max_descendants = num_descendants
            best_parent = p

    return best_parent if best_parent is not None else parents[0]

def get_canonical_ancestors_chain(cell_term):
    """Risale la catena dei padri fino al nodo radice 'cell'."""
    chain = [cell_term]
    current = cell_term

    while True:
        parents = [
            p
            for p in current.superclasses(distance=1, with_self=False)
            if p is not None and p.id and p.name and not p.obsolete
        ]
        if not parents:
            break

        next_parent = (
            parents[0]
            if len(parents) == 1
            else get_most_specific_parent(parents)
        )
        chain.append(next_parent)
        current = next_parent

        if current.id == "CL:0000000":  # cell
            break

    return chain

def get_optimal_canonical_lineage(cell_term, macro_classes = None):
    """
    Reconstructs the optimal lineage for a target cell by combining bottom-up
    """
    if cell_term is None:
        return []

    # only if we call the function outside the cluster one
    if macro_classes is None:
        macro_classes = define_macro_classes()

    # Extract all valid ancestor nodes for the cell term
    ancestors_subgraph = set(cell_term.superclasses(with_self=True))

    matched_macro = None
    for kw in macro_classes:
        kw_lower = kw.lower()
        matches = [
            anc
            for anc in ancestors_subgraph
            if anc.name and kw_lower in anc.name.lower()
        ]
        if matches:
            # Select the top-most / most generic matching node (fewer superclasses)
            matched_macro = min(
                matches, key=lambda a: len(list(a.superclasses()))
            )
            break

    # --- DYNAMIC FALLBACK ---
    # If the cell does not belong to any known macro-class (e.g., 'vascular associated smooth muscle cell')
    if matched_macro is None:
        chain = get_canonical_ancestors_chain(cell_term)
        matched_macro = cell_term  # default fallback
        # Traversal to find the direct child of 'eukaryotic cell' or 'cell'
        for i, node in enumerate(chain):
            if (
                node.name
                and node.name.lower() in ["eukaryotic cell", "cell"]
                and i > 0
            ):
                matched_macro = chain[i - 1]
                break

    # PHASE 2: Top-Down selection based on maximum subtree cardinality
    current_node = matched_macro
    optimal_path = [current_node]

    while current_node != cell_term:
        valid_children = [
            child
            for child in current_node.subclasses(distance=1, with_self=False)
            if child in ancestors_subgraph
        ]

        if not valid_children:
            break

        best_child = (
            valid_children[0]
            if len(valid_children) == 1
            else max(
                valid_children,
                key=lambda c: len(list(c.subclasses(with_self=False))),
            )
        )

        optimal_path.append(best_child)
        current_node = best_child

    # Convert Term objects into name strings
    lineage_names = [
        node.name if node.name else str(node.id) for node in optimal_path
    ]

    # Reverse to maintain standard convention [leaf -> root]
    return list(reversed(lineage_names))

def get_relationships_and_macro(cell_list, ontology = None, macro_classes = None):

    if ontology is None:
        ontology = ms

    if macro_classes is None:
        macro_classes = define_macro_classes()

    relationships = {}
    for cell in cell_list:
        term = return_term_ontology(cell, ontology)
        lineage = get_optimal_canonical_lineage(term, macro_classes)
        relationships[cell] = lineage

    unique_values = {
        item
        for sublist in relationships.values()
        for item in sublist
        if item is not None
    }
    macro_classes_terms = []
    not_found_terms = []

    # Funzione di misura della genericità ontologica
    def get_ontological_depth(term_name: str) -> tuple[int, int]:
        ont_term = return_term_ontology(term_name, ontology)
        if ont_term is not None:
            return (0, len(list(ont_term.superclasses())))
        return (1, len(term_name))

    for class_name in macro_classes:
        matches = [val for val in unique_values if class_name in val.lower()]

        if matches:
            generic_term = min(matches, key=get_ontological_depth)
            macro_classes_terms.append(generic_term)
        else:
            not_found_terms.append(class_name)

    # Add new classes discovered via fallback
    # (eg. 'smooth muscle cell' or 'muscle cell')
    for sublist in relationships.values():
        if sublist:
            root_term = sublist[-1]  # La radice del lineage [leaf -> root]
            if not any(kw.lower() in root_term.lower() for kw in macro_classes):
                if root_term not in macro_classes_terms:
                    macro_classes_terms.append(root_term)

    if len(macro_classes_terms) >= len(macro_classes):
        print("All target terms processed and new dynamic macro classes added!")
    print(f"New terms: {macro_classes_terms}, not present: {not_found_terms}\n")

    return relationships, macro_classes_terms

def cluster_by_macro_class_depth(cell_list, depth_offset= 0, initial_macro_classes = None):
    """
    Clusters cell types within each macro-class based on ontology path depth.
    """

    if initial_macro_classes is None:
        initial_macro_classes = define_macro_classes()

    relationships, macro_classes = get_relationships_and_macro(cell_list=cell_list, ontology=ms, macro_classes=initial_macro_classes)
    clusters_by_macro = defaultdict(lambda: defaultdict(list))

    for cell_name, lineage in relationships.items():
        matched_macro = None
        macro_idx = None

        # 1. Scan the lineage FROM TOP TO BOTTOM (from root 'cell' towards leaf)
        # Temporarily reverse the list to find the most general ontology node first
        reversed_lineage = list(reversed(lineage))
        total_depth = len(lineage)

        for rev_idx, node_name in enumerate(reversed_lineage):
            node_lower = node_name.lower()

            # Check keywords in defined priority order
            for kw in macro_classes:
                if kw.lower() in node_lower:
                    matched_macro = kw  # Use the canonical keyword
                    # Convert the reversed index back to the original lineage index
                    macro_idx = total_depth - 1 - rev_idx
                    break

            if matched_macro is not None:
                break  # Highest macro node in the tree found

        # 2. If a macro-class was matched
        if macro_idx is not None:
            # Calculate sub-cluster depth based on depth_offset
            target_idx = max(0, macro_idx - depth_offset)
            cluster_key = lineage[target_idx]

            clusters_by_macro[matched_macro][cluster_key].append(cell_name)

    return {
        macro: dict(clusters) for macro, clusters in clusters_by_macro.items()
    }

def print_clusters(cluster_dict, initial_cells):

    print("HIERARCHICAL CELL CLUSTERS BY ONTOLOGY")

    count = 0
    for new_macro_class, clusters in cluster_dict.items():
        # Count total cells and total clusters within this macro-class
        total_cells = sum(len(cell_list_new) for cell_list_new in clusters.values())
        class_cluster = len(clusters)
        count += total_cells

        print(
            f"\n[MACRO-CLASS]: '{new_macro_class}' ( {total_cells} cells, clustered in {class_cluster} clusters )"
        )
        print("-" * 50)

        for cluster_node, cells in clusters.items():
            print(f"  └─ Cluster Node: '{cluster_node}' ({len(cells)} cells)")
            for cell in cells:
                print(f"       • {cell}")

    # Sanity check for missing cells
    correctly_clustered = set()

    for _, clusters in cluster_dict.items():
        for cluster_node, cell_list_new in clusters.items():
            correctly_clustered.update(cell_list_new)

    not_clustered = set(initial_cells) - correctly_clustered
    print(f"\nMissing cells ({len(not_clustered)}): {not_clustered if len(not_clustered) > 0 else None}")

    return

def remove_prefix(cell_type: str) -> str:
    return cell_type.removeprefix('q05cell_abundance_w_sf_')

def build_cell_type_mapping(
    cell_list,
    offset=0,
    initial_macro_classes=None,
    macro_merge={},
    merge_similar=True,
):
    """builds a dictionary mapping original cell types to target classes.

    returns the cell_mapping dict and a list of unmapped cell types (e.g.
    'multiplets').
    """
    # strict validation of macro_merge keys
    if initial_macro_classes is not None:
        is_valid_dict = isinstance(macro_merge, dict) and (
            len(set(initial_macro_classes) & set(macro_merge.keys()))
            == len(set(initial_macro_classes))
        )

        if not is_valid_dict:
            print(
                "WARNING: Since you didn't include how to merge your macro classes, "
                "the algorithm restored automatic-defined macro-classes to perform the mapping."
            )
            initial_macro_classes = None
            macro_merge = True

    if initial_macro_classes is None:
        initial_macro_classes = define_macro_classes()

    # run ontology-based clustering
    cluster_dict = cluster_by_macro_class_depth(
        cell_list=cell_list,
        depth_offset=offset,
        initial_macro_classes=initial_macro_classes,
    )

    cell_mapping = {}

    # case 1: macro-level mapping (offset == 0)
# case 1: macro-level mapping (offset == 0)
    if offset == 0:
        for macro_class, clusters in cluster_dict.items():
            if isinstance(macro_merge, dict):
                target_label = macro_merge.get(macro_class, macro_class)
            else:
                target_label = macro_class

            for cluster_node, cells in clusters.items():
                for cell in cells:
                    cell_mapping[cell] = target_label

    # case 2: fine-grained interactive mapping (offset >= 1)
    else:
        candidates = {
            macro: list(clusters.keys())
            for macro, clusters in cluster_dict.items()
            if len(clusters) > 1
        }

        if merge_similar and candidates:
            macro_names_str = ", ".join(candidates.keys())
            print(
                f"\nFound {len(candidates)} possible cluster merges for macro-classes: {macro_names_str}\n"
            )

        for macro_class, clusters in cluster_dict.items():
            cluster_names = list(clusters.keys())
            cluster_rename_map = {name: name for name in cluster_names}

            if merge_similar and len(cluster_names) > 1:
                # TODO: introduce image-feature-space-based similarity
                formatted_clusters = ", ".join(
                    [
                        f"'{name}' ({idx + 1})"
                        for idx, name in enumerate(cluster_names)
                    ]
                )
                print(f"\n[MACRO-CLASS]: '{macro_class}'")
                print(f"Clusters: [{formatted_clusters}]")
                print("Please type:")
                print(f"  *       -> \t\t to cluster them all into '{macro_class}' macro-class")
                print('  n1,n2,...,nk,name -> \t\t number combination and new name to group them\n'
                    '(e.g. if you want to group clusters 1 and 3 into the "super endothelial" type 1,3,super endothelial and press enter)')
                print("  [Enter] -> \t\t keep all clusters separated as they are")

                user_input = input("Selection: ").strip()

                if user_input == "*":
                    for name in cluster_names:
                        cluster_rename_map[name] = macro_class

                elif user_input:
                    parts = [p.strip() for p in user_input.split(",")]
                    if len(parts) >= 2:
                        try:
                            indices = [int(idx) - 1 for idx in parts[:-1]]
                            new_name = parts[-1]

                            for idx in indices:
                                if 0 <= idx < len(cluster_names):
                                    original_cluster = cluster_names[idx]
                                    cluster_rename_map[
                                        original_cluster
                                    ] = new_name
                        except ValueError:
                            print(
                                "Invalid input format. Keeping default cluster assignments."
                            )

            for cluster_node, cells in clusters.items():
                final_label = cluster_rename_map[cluster_node]
                for cell in cells:
                    cell_mapping[cell] = final_label

    # track unmapped cells (cells in cell_list that are not in cell_mapping)
    unmapped_cells = [cell for cell in cell_list if cell not in cell_mapping]

    if unmapped_cells:
        print(f"\nUnmapped/Excluded terms (no ontology lineage): {unmapped_cells}")

    return cell_mapping, unmapped_cells

# TODO

def mapping_function(
    df, cell_list, offset=0, macro_merge=None, merge_similar=True
):
    """builds cell type mapping automatically, applies it to abundance columns [4:],

    drops unmapped columns (e.g. multiplets), aggregates merged classes,
    re-normalizes rows (l1 norm) if any column was discarded, and re-attaches
    metadata columns [:4].
    """
    if macro_merge is None:
        macro_merge = {}

    # 1. separate metadata columns from abundance columns
    df_meta = df.iloc[:, :4].copy()
    df_abundance = df.iloc[:, 4:].copy()

    # 2. strip prefix unconditionally from abundance columns
    df_abundance.columns = [remove_prefix(col) for col in df_abundance.columns]

    # 3. generate cell type mapping and identify unmapped terms
    cell_mapping, unmapped_cells = build_cell_type_mapping(
        cell_list=cell_list,
        offset=offset,
        macro_merge=macro_merge,
        merge_similar=merge_similar,
    )

    # 4. identify valid abundance columns present in mapping
    valid_cols = [col for col in df_abundance.columns if col in cell_mapping]
    num_dropped = len(df_abundance.columns) - len(valid_cols)

    # 5. filter out unmapped columns
    df_mapped = df_abundance[valid_cols].copy()

    # 6. rename columns to target classes and aggregate duplicates (sum abundances)
    df_mapped = df_mapped.rename(columns=cell_mapping)
    df_mapped = df_mapped.groupby(level=0, axis=1).sum()

    # 7. l1 re-normalization if abundance columns were discarded
    if num_dropped > 0:
        print(
            f"dropped {num_dropped} unmapped column(s) {unmapped_cells}. performing l1 re-normalization..."
        )
        row_sums = df_mapped.sum(axis=1)
        # safe division to avoid division by zero
        df_mapped = df_mapped.div(row_sums.replace(0, 1), axis=0)

    # 8. concatenate preserved metadata back with mapped abundance features
    df_final = pd.concat([df_meta, df_mapped], axis=1)

    return df_final, cell_mapping, unmapped_cells

# i want to be able to use it also outside
def get_default_macro_category(col_name, ontology_dict = None):
    """Maps a cell term or intermediate class to its corresponding macro-category.
    Uses the canonical ontology lineage to identify the root category.
    """
    col_lower = col_name.lower()

    # 1. Retrieve the ontological lineage if available
    lineage = []
    if ontology_dict and col_name in ontology_dict:
        # Use pre-calculated lineage from dictionary
        lineage = ontology_dict[col_name]
    else:
        # Fallback to querying the ontology directly
        try:
            lineage = get_optimal_canonical_lineage(col_lower)
        except Exception:
            lineage = []

    # 2. Extract root term from lineage if available, otherwise use the column name itself
    terms_to_check = [col_lower]
    if lineage and len(lineage) > 0:
        # Include both lineage nodes (root is last) and the original column name
        terms_to_check = [term.lower() for term in lineage] + [col_lower]

    # 3. Perform matching based on lineage terms with priority ordering
    for term in terms_to_check:
        if "neoplastic" in term or "malignant" in term:
            return "Neoplastic"

    for term in terms_to_check:
        if any(
            k in term
            for k in [
                "leukocyte",
                "leucocyte",
                "inflammatory",
                "immune",
                "lymphocyte",
                "mononuclear phagocyte",
                "myeloid",
            ]
        ):
            return "Inflammatory"

    for term in terms_to_check:
        if "epithelial" in term:
            return "Epithelial"

    # Fallback for everything else
    return "Connective and Soft tissues"

def map_to_4_macro_classes(df, custom_mapping=None, fill_missing=True):
    """Maps fine/intermediate abundance columns [:4 meta, 4: features] back to 4 canonical macro-classes.

    Fallback rules (if custom_mapping is None or incomplete):
    - 'neoplastic' / 'malignant' -> 'neoplastic'
    - 'leukocyte' / 'inflammatory' / 'leucocyte' / 'immune' -> 'inflammatory'
    - 'epithelial' -> 'epithelial'
    - Everything else -> 'connective and soft tissues'
    """
    df_meta = df.iloc[:, :4].copy()
    df_abundance = df.iloc[:, 4:].copy()
    abundance_cols = list(df_abundance.columns)

    canonical_targets = [
        "Neoplastic",
        "Inflammatory",
        "Epithelial",
        "Connective and Soft tissues",
    ]

    # Default rule based on keyword matching

    # Check if a valid and complete custom_mapping dictionary was provided
    use_custom = False
    if isinstance(custom_mapping, dict):
        # Verify if all abundance columns in the dataframe are present in custom_mapping
        missing_keys = [
            col for col in abundance_cols if col not in custom_mapping
        ]
        if len(missing_keys) == 0:
            use_custom = True
        else:
            print(
                f"WARNING: custom_mapping is incomplete. Missing keys for columns: {missing_keys}. "
                f"Falling back to default keyword matching."
            )

    # Construct final column renaming dictionary
    if use_custom:
        final_column_mapping = {col: custom_mapping[col] for col in abundance_cols}
    else:
        final_column_mapping = {
            col: get_default_macro_category(col) for col in abundance_cols
        }

    # Aggregate columns (sum abundances per target class)
    df_macro = df_abundance.rename(columns=final_column_mapping)
    df_macro = df_macro.groupby(level=0, axis=1).sum()

    # Handle missing target classes (e.g. epithelial)
    if fill_missing:
        for target in canonical_targets:
            if target not in df_macro.columns:
                df_macro[target] = 0.0
        # Ensure consistent column ordering across datasets
        df_macro = df_macro[canonical_targets]

    # Re-attach preserved metadata
    df_final = pd.concat([df_meta, df_macro], axis=1)

    return df_final