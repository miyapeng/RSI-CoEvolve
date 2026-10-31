"""Dense HF structure validation and bounded tensor probes; no fabricated model update."""

import hashlib
import json
import struct
from pathlib import Path

from .records import digest, read, sha, write


def complete_dense_export(source, destination):
    """Carry frozen vision/MTP tensors forward without ever filling missing LM weights.

    Slime's dense text trainer exports the updated language model. Auxiliary tensors
    must occupy their own shard: linking a source shard could expose stale LM tensors
    to loaders that enumerate every safetensors file instead of consulting the index.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Export must not overwrite its source checkpoint")
    origin = read(source / "model.safetensors.index.json")["weight_map"]
    index = read(destination / "model.safetensors.index.json")
    missing = set(origin) - set(index["weight_map"])
    if set(index["weight_map"]) - set(origin) or any(
        not name.startswith(("model.visual.", "mtp.")) for name in missing
    ):
        raise ValueError("Dense export has missing language-model or unexpected tensors")
    tensors = {}
    for shard in sorted({origin[name] for name in missing}):
        with safe_open(source / shard, framework="pt", device="cpu") as f:
            for name in sorted(missing):
                if origin[name] == shard:
                    tensors[name] = f.get_tensor(name)
    if tensors:
        filename = "frozen-auxiliary.safetensors"
        if (destination / filename).exists():
            raise ValueError("Incomplete earlier auxiliary export; inspect before resuming")
        save_file(tensors, destination / filename, metadata={"format": "pt"})
        index["weight_map"].update({name: filename for name in tensors})
        index["metadata"]["total_size"] += sum(t.numel() * t.element_size() for t in tensors.values())
        write(destination / "model.safetensors.index.json", index)
    return {"frozen_auxiliary_tensors_copied": len(tensors), "source_checkpoint": str(source)}


def inspect(checkpoint):
    root = Path(checkpoint).resolve()
    config = json.loads((root / "config.json").read_text())
    tc = config.get("text_config", config)
    if config.get("model_type") != "qwen3_5" or tc.get("hidden_size") != 4096 or tc.get("num_hidden_layers") != 32:
        raise ValueError("This training configuration is specifically dense Qwen3.5-9B")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    found = {}
    probes = {}
    sizes = {}
    for filename in sorted(set(index["weight_map"].values())):
        file = root / filename
        with file.open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
            last = 0
            for name, info in sorted(
                ((k, v) for k, v in header.items() if k != "__metadata__"), key=lambda kv: kv[1]["data_offsets"][0]
            ):
                start, end = info["data_offsets"]
                if start != last or end < start:
                    raise ValueError("Invalid safetensors data offsets")
                last = end
                found[name] = filename
                # Read small, deterministic ranges from each tensor, not the entire shared model.
                h = hashlib.sha256()
                n = min(256, end - start)
                for offset in sorted({start, max(start, (start + end - n) // 2), end - n}):
                    f.seek(8 + header_size + offset)
                    h.update(f.read(n))
                probes[name] = h.hexdigest()
            if file.stat().st_size != 8 + header_size + last:
                raise ValueError("Truncated or trailing safetensors payload")
            sizes[filename] = last
    if found != index["weight_map"] or sum(sizes.values()) != index["metadata"]["total_size"]:
        raise ValueError("Safetensors index does not cover the available weights")
    tokenizer = json.loads((root / "tokenizer_config.json").read_text())
    if not (root / "tokenizer.json").exists() or "<tool_call>" not in tokenizer.get("chat_template", ""):
        raise ValueError("Missing tokenizer or Qwen tool-call chat template")
    return {
        "checkpoint": str(root),
        "config_sha256": sha(root / "config.json"),
        "index_sha256": sha(root / "model.safetensors.index.json"),
        "tokenizer_config_sha256": sha(root / "tokenizer_config.json"),
        "tensor_probe_sha256": digest(probes),
        "tensor_probes": probes,
        "shards": len(sizes),
        "eos_token_id": tc.get("eos_token_id"),
        "max_position_embeddings": tc.get("max_position_embeddings"),
        "validation": "headers, sizes, index, tokenizer and sparse tensor probes; not a full payload checksum",
    }
