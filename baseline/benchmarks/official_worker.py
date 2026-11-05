"""Subprocess entry: import the pinned native harness before any globally installed copy."""

import asyncio
import contextlib
import importlib
import json
import logging
import os
import sys
from importlib.metadata import distributions
from pathlib import Path


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    repo = Path(request["source"]["repo"]).resolve()
    sys.path[:0] = [str(repo), str(Path(__file__).resolve().parents[2])]
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO)
    from baseline.benchmarks import get_benchmark
    from baseline.records import write

    adapter = get_benchmark({"benchmark": request["benchmark"]})
    # Recheck source immediately before import, not just at command planning time.
    adapter.source({"official": {"repo": str(repo)}})
    write(
        Path(request["output"]) / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "source": request["source"],
            "adapter_code_hash": request["adapter_code_hash"],
            "packages": {d.metadata["Name"]: d.version for d in distributions()},
        },
    )
    module = importlib.import_module(adapter.__module__)
    if adapter.name == "vision2web":
        with contextlib.ExitStack() as stack:
            if request["options"].get("transport", "docker") == "tencent_e2b":
                from baseline.benchmarks.vision2web_e2b import transport

                stack.enter_context(transport(request))

            async def run():
                return await asyncio.wait_for(
                    module.run_native(request), timeout=request["options"].get("episode_timeout", 14400)
                )

            result = asyncio.run(run())
    else:
        result = module.run_native(request)
    write(Path(request["output"]) / "native_result.json", result)


if __name__ == "__main__":
    main()
