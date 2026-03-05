# Dataset Schema & Pipeline Outputs (Preprocess)

This document defines the schema contracts for each preprocessing stage and the final snapshot dataset.

## Overview

Pipeline stages (in order):

1. `canonicalize`  → `01_canonicalize/`
2. `minhash`       → `02_minhash/`
3. `pairs`         → `03_pairs/`
4. `cluster_map`   → `04_cluster_map/`
5. `snapshot`      → `05_snapshot/` (final deduplicated dataset)

Each stage reads Parquet and writes Parquet.

---

## Raw Input Dataset (Required)

**Location:** `run.input_dir`  
**Format:** Parquet

### Required columns
| Column | Type | Description |
|---|---|---|
| `id` | string | Unique example identifier |
| `dataset` | string | Source dataset name (if available) |
| `language` | string | Language label, e.g. `"python"` |
| `has_code` | bool | Whether `code_ref` contains code |
| `instruction` | string | Instruction / prompt text |
| `code_ref` | string | Code snippet / completion |

### Optional columns (pass-through)
Any additional metadata columns are allowed and may be passed through stages.

---

## Stage 01 — Canonicalize (`01_canonicalize/`)

**Purpose:** Parse Python code and produce a canonical representation for similarity detection.

### Output columns (adds)
| Column | Type | Description |
|---|---|---|
| `parse_ok` | bool | Whether parsing/canonicalization succeeded |
| `parse_err` | string or null | Failure reason |
| `node_types` | list[string] | Canonical AST node-type sequence (when `representation: node_types`) |
| `canon_dump` | string | Canonical AST dump (when `representation: dump`) |

### Notes
- Non-Python or rows with `has_code=false` are marked `parse_ok=false`.
- Downstream stages typically filter to `parse_ok=true`.

---

## Stage 02 — MinHash (`02_minhash/`)

**Purpose:** Compute MinHash signatures from canonical representation (currently node_types shingles).

### Output columns (adds)
| Column | Type | Description |
|---|---|---|
| `sig_ok` | bool | Whether a valid signature was computed |
| `sig` | list[int] | MinHash signature length `k` |

### Notes
- Uses shingles of length `shingle_n`.
- Rows with `parse_ok=false` or too-short `node_types` are `sig_ok=false`.

---

## Stage 03 — Pairs (`03_pairs/`)

**Purpose:** Generate candidate near-duplicate pairs using LSH banding.

### Output schema
| Column | Type | Description |
|---|---|---|
| `id1` | string | Example id (endpoint A) |
| `id2` | string | Example id (endpoint B) |

### Notes
- Buckets are capped by `max_bucket_size`.
- Pair generation per bucket is capped by `max_pairs_per_bucket` (deterministic order).

---

## Stage 04 — Cluster Map (`04_cluster_map/`)

**Purpose:** Build connected components (clusters) over candidate pairs.

### Output schema
| Column | Type | Description |
|---|---|---|
| `id` | string | Example id appearing in at least one pair |
| `cluster_id` | string | Stable cluster identifier |

### Notes
- `cluster_id` stability: computed as `md5(min_id_in_component)` (stable across runs).
- IDs that never appear in any pair will not be present in `cluster_map` (handled in snapshot via singleton fill).

---

## Stage 05 — Snapshot (Final) (`05_snapshot/`)

**Purpose:** Produce a deduplicated snapshot of the dataset.

### Output columns (default keep set)
| Column | Type | Description |
|---|---|---|
| `dataset` | string | Source dataset |
| `id` | string | Unique id |
| `instruction` | string | Instruction text |
| `code_ref` | string | Code |
| `language` | string | Language |
| `has_code` | bool | Has code |
| `cluster_id` | string | Cluster id (singleton-filled) |

### Dedup rule
- Group key: `cluster_id` (cluster mode)
- Representative selection (deterministic):
  1. longest `code_ref`
  2. then longest `instruction`
  3. then smallest `md5(id)`

### Singleton handling
If `cluster_id` is missing after join, it is filled as:
- `cluster_id = md5(id)`

---

## Configuration References

Key config sections:
- `canonicalize`: canonicalization knobs
- `minhash`: `shingle_n`, `num_perm`, `seed0`
- `pairs`: `k`, `b`, `max_bucket_size`, `max_pairs_per_bucket`
- `snapshot`: `mode`, `key_col`, `snapshot_keep_cols` (if configured)

---

## Invariants & Guarantees

- All outputs are Parquet.
- IDs are treated as strings.
- Representative selection is deterministic.
- No stage requires materializing entire datasets on the driver (`take_all()` is avoided).

---

## Known Limitations

- Windows Ray may have logging/iterator stability issues; Linux/WSL recommended for large runs.
- Very large candidate pair graphs may still stress driver memory during Union-Find; tune LSH caps accordingly.