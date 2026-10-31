"""Small, durable experiment records. Never serialize clients or environment variables."""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIME_COMMIT = "0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e"
ORCHARD_COMMIT = "bd8da32b3a075128700caab75bbc491fb6fdc16e"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        os.chmod(tmp, 0o600)
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        os.chmod(path, 0o600)
        f.write(json.dumps(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read(path):
    return json.loads(Path(path).read_text())
