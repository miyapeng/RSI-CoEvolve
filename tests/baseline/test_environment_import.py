import importlib.util
import subprocess

from baseline.records import ROOT


def test_image_copy_keeps_nested_modules_and_preserves_conda_tools(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "image_environment", ROOT / "scripts/baseline/import_image_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source, target = tmp_path / "source", tmp_path / "target"
    paths = [
        "setuptools/__init__.py",
        "torch/distributed/pipelining/__init__.py",
        "scikit_build_core/setuptools/__init__.py",
    ]
    for name in paths:
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("image package")
    original = target / paths[0]
    original.parent.mkdir(parents=True)
    original.write_text("conda package")
    subprocess.run(
        [
            "rsync",
            "-a",
            "--ignore-existing",
            *[f"--exclude={x}" for x in module.COPY_EXCLUDES],
            str(source) + "/",
            str(target) + "/",
        ],
        check=True,
    )
    assert original.read_text() == "conda package"
    assert all((target / name).read_text() == "image package" for name in paths[1:])
