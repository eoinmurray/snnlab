"""Detached submission and durable collection at mocked Modal service boundaries."""

import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    class Image:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    class App:
        def __init__(self, *args):
            self.contexts = []

        def function(self, **kwargs):
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

        def run(self, **kwargs):
            self.contexts.append(kwargs)
            return nullcontext()

    class Volume:
        def __init__(self):
            self.files = {}
            self.commits = 0
            self.reloads = 0

        def read_file(self, name):
            if name not in self.files:
                raise FileNotFoundError(name)
            yield self.files[name]

        def commit(self):
            self.commits += 1

        def reload(self):
            self.reloads += 1

    volume = Volume()
    sdk = SimpleNamespace(
        App=App,
        Volume=SimpleNamespace(from_name=lambda *a, **k: volume),
        Image=SimpleNamespace(debian_slim=lambda **k: Image()),
        is_local=lambda: True,
        enable_output=nullcontext,
    )
    monkeypatch.setitem(sys.modules, "modal", sdk)
    module = importlib.import_module("benchmarks.shd.benchmark")
    module = module.modal_launcher(local_root=tmp_path, data_root=tmp_path / "remote")
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/shd"))
    return module


def test_python_launch_is_detached_and_returns_after_submission(launcher, monkeypatch):
    calls = []

    def spawn(*args):
        calls.append(args)
        return SimpleNamespace(object_id="fc-persisted")  # No get()/remote() method.

    monkeypatch.setattr(launcher.worker, "spawn", spawn, raising=False)
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--trace-mode", "both"])
    ticket = launcher.cli()
    assert launcher.app.contexts == [{"detach": True}]
    assert len(calls) == 1
    assert calls[0][3] == "both"
    assert ticket["call_id"] == "fc-persisted"
    path = launcher.LOCAL_ROOT / "benchmarks/shd/results" / "cloud.json"
    assert json.loads(path.read_text()) == ticket


def test_collect_reads_volume_without_app_context_or_gpu(launcher, monkeypatch):
    run_id = "shd-test"
    cloud = {
        **launcher.cloud_paths(run_id),
        "status": "complete",
        "report_files": ["summary.json", "tracing/summary.json"],
    }
    launcher.volume.files = {
        f"/runs/{run_id}/cloud.json": json.dumps(cloud).encode(),
        f"/runs/{run_id}/summary.json": b'{"completed_updates": 290}',
        f"/runs/{run_id}/tracing/summary.json": b'{"status": "complete"}',
    }

    def forbidden(*args, **kwargs):
        pytest.fail("collection must not start a worker")

    monkeypatch.setattr(launcher.worker, "spawn", forbidden, raising=False)
    monkeypatch.setattr(
        sys, "argv", ["benchmark.py", "--mode", "collect", "--run-id", run_id]
    )
    destination = launcher.LOCAL_ROOT / "benchmarks/shd/results"
    destination.mkdir(parents=True)
    (destination / "cloud.json").write_text(
        json.dumps({"call_id": "fc-original", "run_id": run_id})
    )
    assert launcher.cli() == destination
    assert launcher.app.contexts == []
    assert (
        json.loads((destination / "summary.json").read_text())["completed_updates"]
        == 290
    )
    assert (
        json.loads((destination / "cloud.json").read_text())["call_id"] == "fc-original"
    )
    assert (destination / "tracing/summary.json").exists()


def test_worker_persists_completion_manifest_without_local_collector(
    launcher, monkeypatch
):
    def run(cfg, output, *args, **kwargs):
        assert cfg.epochs == 10
        assert cfg.batch_size == 256
        output.mkdir(parents=True)
        for name in (
            "configuration.json",
            "metrics.json",
            "timing.json",
            "summary.json",
        ):
            (output / name).write_text("{}")

    monkeypatch.setitem(
        sys.modules,
        "benchmark",
        SimpleNamespace(
            run=run, Config=importlib.import_module("benchmarks.shd.benchmark").Config
        ),
    )
    result = launcher.worker("shd-cloud", 10, False)
    manifest = launcher.DATA_ROOT / "runs/shd-cloud/cloud.json"
    saved = json.loads(manifest.read_text())
    assert result == {"run_id": "shd-cloud", "status": "complete"}
    assert saved["status"] == "complete"
    assert saved["checkpoint"] == "runs/shd-cloud/final.checkpoint"
    assert len(saved["report_files"]) == 4
    assert launcher.volume.reloads == launcher.volume.commits == 1


def test_worker_commits_partial_artifacts_and_propagates_failure(launcher, monkeypatch):
    def run(cfg, output, *args, **kwargs):
        output.mkdir(parents=True)
        (output / "configuration.json").write_text("{}")
        raise RuntimeError("injected worker failure")

    monkeypatch.setitem(
        sys.modules,
        "benchmark",
        SimpleNamespace(
            run=run, Config=importlib.import_module("benchmarks.shd.benchmark").Config
        ),
    )
    with pytest.raises(RuntimeError, match="injected worker failure"):
        launcher.worker("shd-failure", 10, False)
    assert launcher.volume.commits == 1
    assert not (launcher.DATA_ROOT / "runs/shd-failure/cloud.json").exists()
    assert (launcher.DATA_ROOT / "runs/shd-failure/configuration.json").exists()


@pytest.mark.parametrize("run_id", ["", "../escape", "/absolute", "has spaces"])
def test_invalid_run_ids_rejected(launcher, run_id):
    with pytest.raises(ValueError, match="run ID"):
        launcher.collect(run_id)


def test_collection_rejects_unsafe_manifest_before_writing_reports(launcher):
    launcher.volume.files["/runs/shd-bad/cloud.json"] = json.dumps(
        {"status": "complete", "run_id": "shd-bad", "report_files": ["../outside.json"]}
    ).encode()
    with pytest.raises(ValueError, match="invalid report path"):
        launcher.collect("shd-bad")
    assert not (launcher.LOCAL_ROOT / "outside.json").exists()


def test_unfinished_run_is_not_reported_as_complete(launcher):
    with pytest.raises(RuntimeError, match="No completion manifest"):
        launcher.collect("shd-not-complete")
    assert launcher.app.contexts == []


def test_worker_module_import_does_not_assume_local_launcher_depth(
    tmp_path, monkeypatch
):
    class App:
        def function(self, **kwargs):
            return lambda fn: fn

        def local_entrypoint(self):
            return lambda fn: fn

    sdk = SimpleNamespace(
        App=lambda *args: App(),
        Volume=SimpleNamespace(from_name=lambda *args, **kwargs: object()),
        is_local=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "modal", sdk)
    module = importlib.import_module("benchmarks.shd.benchmark")
    assert Path("/workspace") == module.modal_launcher().LOCAL_ROOT
