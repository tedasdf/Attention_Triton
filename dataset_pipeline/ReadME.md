Deliverable 1: One Ray pipeline script
Suggested file layout
dataset_pipeline/
  pipeline.py              # main entry (Ray Data)
  stages/
    normalize.py
    canonicalize.py
    minhash.py
    lsh.py
    quality.py
    snapshot.py
    split.py
  config/
    pipeline.yaml
  utils/
    io.py
    hashing.py
    logging.py
Required outputs

snapshot parquet shards

split parquet shards

manifests (jsonl) for snapshot + each split

run metadata:

run_info.json (git commit, dvc remote, timestamp, ray version, params)

Deliverable 2: Versioning + DVC publish script
What it should do

Compute a new snapshot_id (v0001, v0002, …)

Run the pipeline (or verify pipeline already ran)

dvc add the output directories:

snapshots/unified_python/vX

splits/unified_python/vX

git add the generated .dvc files (or updated dvc.yaml/dvc.lock)

dvc push -r mys3remote

git commit and push to GitHub

Tag the commit:

git tag dataset-v0001 (optional)

Git ignore policy

keep big data out of Git:

ignore snapshots/**, splits/** (except manifests if you want them in Git)

track via DVC.