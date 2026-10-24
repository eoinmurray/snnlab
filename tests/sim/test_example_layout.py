"""Every example replaces its own results without touching sibling files."""

import importlib.util
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
DIRECTORIES = sorted(path for path in EXAMPLES.iterdir() if path.is_dir())


@pytest.mark.parametrize("directory", DIRECTORIES, ids=lambda path: path.name)
def test_results_are_replaced_and_siblings_preserved(directory, tmp_path, monkeypatch):
    assert sorted(path.name for path in directory.glob("*.py")) == ["example.py"]
    spec = importlib.util.spec_from_file_location(
        f"layout_{directory.name.replace('-', '_')}", directory / "example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = tmp_path / "results"
    (results / "old.bundle").mkdir(parents=True)
    (results / "old.bundle" / "stale.json").write_text("old run")
    sibling = tmp_path / "example.py"
    sibling.write_text("source")
    monkeypatch.setattr(module, "OUTPUT_DIR", results)
    module.prepare_results()
    assert results.is_dir()
    assert list(results.iterdir()) == []
    assert sibling.read_text() == "source"
    (results / "new.json").write_text("new run")
    module.prepare_results()
    assert list(results.iterdir()) == []


def test_results_symlink_does_not_delete_target(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "layout_quickstart", EXAMPLES / "quickstart/example.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "other-results"
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    results = tmp_path / "results"
    results.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(module, "OUTPUT_DIR", results)
    module.prepare_results()
    assert not results.is_symlink()
    assert list(results.iterdir()) == []
    assert (target / "keep.txt").read_text() == "keep"
