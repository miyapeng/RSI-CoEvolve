"""Fetch the official anonymous key/catalog, without starting a game or model call."""

import argparse
import json
import os
from pathlib import Path

import httpx
from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, required=True, help="Private .env file outside the repository")
    parser.add_argument("--output", type=Path, required=True, help="Catalog JSON file")
    args = parser.parse_args()
    key_file = args.key_file.expanduser().resolve()
    if key_file.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("Keep ARC credentials outside the repository")
    with httpx.Client(base_url="https://arcprize.org", timeout=30) as client:
        if key_file.exists():
            if key_file.stat().st_mode & 0o077:
                parser.error("Existing credential file must be owner-only (chmod 600)")
            key = dotenv_values(key_file).get("ARC_API_KEY")
            if not key:
                parser.error("Existing credential file has no ARC_API_KEY; it was not modified")
        else:
            response = client.get("/api/games/anonkey")
            response.raise_for_status()
            key = response.json()["api_key"]
            if not isinstance(key, str) or not key or not all(c.isalnum() or c in "_-" for c in key):
                raise ValueError("Unexpected anonymous key format")
            key_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with os.fdopen(os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
                f.write(f"ARC_API_KEY={key}\n")
        response = client.get("/api/games", headers={"X-API-Key": key})
        response.raise_for_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(response.json(), indent=2) + "\n")
    print(f"Catalog saved to {args.output}; credential retained in {key_file} (not printed)")
    print("Catalog visibility does not imply anonymous play access to every game.")


if __name__ == "__main__":
    main()
