"""Coverage for pure runlog helpers not hit by the integration tests:
the human-readable formatters + provenance/file-listing helpers.
"""

from __future__ import annotations

from snnlab.sim import runlog

# ── runlog.py formatters + helpers ────────────────────────────────────────


def test_format_eta_covers_all_ranges():
    assert runlog.format_eta(45) == "45s"
    assert runlog.format_eta(510) == "8m30s"
    assert runlog.format_eta(4320) == "1h12m"


def test_format_bytes_covers_all_units():
    assert runlog.format_bytes(512) == "512 B"
    assert runlog.format_bytes(2048).endswith("KB")
    assert runlog.format_bytes(5 * 1024**2).endswith("MB")
    assert runlog.format_bytes(3 * 1024**3).endswith("GB")
    assert runlog.format_bytes(2 * 1024**4).endswith("TB")


def test_provenance_returns_expected_keys():
    # provenance() calls _git_sha() and _env_hash() internally.
    p = runlog.provenance()
    assert {"git_sha", "python_env_hash", "run_id", "torch_version"} <= p.keys()


def test_list_output_files_lists_written_files(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.npz").write_bytes(b"\x00\x01")
    files = runlog.list_output_files(tmp_path)  # list of (name, size_bytes) tuples
    names = {name for name, _ in files}
    assert {"a.json", "b.npz"} <= names
