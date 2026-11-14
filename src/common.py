"""Shared helpers: locating the generated dataset and streaming JSON / JSONL records."""

import glob
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "ecommerce")
HBASE_HOST = os.environ.get("HBASE_HOST", "localhost")
HBASE_PORT = int(os.environ.get("HBASE_PORT", "9090"))


def entity_files(data_dir, entity):
    """Return the files for an entity, e.g. 'sessions' -> sessions_0.jsonl, sessions_1.jsonl ..."""
    if entity == "sessions":
        files = glob.glob(os.path.join(data_dir, "sessions_*.json*"))
        return sorted(files, key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))
    for ext in ("jsonl", "json"):
        path = os.path.join(data_dir, f"{entity}.{ext}")
        if os.path.exists(path):
            return [path]
    raise FileNotFoundError(f"No {entity} file in {data_dir}. Run: python src/generate_data.py")


def iter_records(data_dir, entity, limit=None):
    """Stream records from JSON Lines (preferred) or a JSON array file without loading everything."""
    n = 0
    for path in entity_files(data_dir, entity):
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                rows = (json.loads(line) for line in f if line.strip())
            else:
                rows = iter(json.load(f))
            for row in rows:
                yield row
                n += 1
                if limit is not None and n >= limit:
                    return


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
