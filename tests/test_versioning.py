"""Release preparation must preserve consistent metadata and historical notes."""

import runpy
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = runpy.run_path(str(ROOT / "scripts/version.py"))


@pytest.fixture
def release_tree(tmp_path):
    for relative in (
        "pyproject.toml",
        "uv.lock",
        "CHANGELOG.md",
        "src/snnlab/__init__.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / relative).read_text())
    return tmp_path


@pytest.mark.parametrize(
    "part, expected", [("patch", "0.1.1"), ("minor", "0.2.0"), ("major", "1.0.0")]
)
def test_preparation_preserves_history_and_synchronizes_versions(
    release_tree, part, expected
):
    original = (release_tree / "CHANGELOG.md").read_text()
    version, changes = HELPER["prepare"](release_tree, part, date(2026, 10, 4))
    assert version == expected
    assert (release_tree / "CHANGELOG.md").read_text() == original
    for path, content in changes.items():
        path.write_text(content)
    assert HELPER["check"](release_tree, f"v{expected}") == expected
    changelog = (release_tree / "CHANGELOG.md").read_text()
    assert f"## [Unreleased]\n\n## [{expected}] - 2026-10-04" in changelog
    assert original[original.index("## [0.1.0]") :] in changelog
    assert changelog.count("Six runnable general examples") == 1


def test_rejects_duplicate_lock_version_and_wrong_tag(release_tree):
    with pytest.raises(ValueError, match="Expected tag"):
        HELPER["check"](release_tree, "v9.9.9")
    lock = release_tree / "uv.lock"
    lock.write_text(
        lock.read_text().replace(
            'name = "snnlab"\n', 'name = "snnlab"\nversion = "0.1.1"\n'
        )
    )
    with pytest.raises(ValueError, match="uv.lock"):
        HELPER["check"](release_tree)


def test_refuses_empty_release_and_backdated_release(release_tree):
    with pytest.raises(ValueError, match="cannot precede"):
        HELPER["prepare"](release_tree, "patch", date(2026, 10, 2))
    path = release_tree / "CHANGELOG.md"
    text = path.read_text()
    path.write_text(
        text[: text.index("## [Unreleased]")]
        + "## [Unreleased]\n\n### Added\n\n"
        + text[text.index("## [0.1.0]") :]
    )
    with pytest.raises(ValueError, match="no Unreleased notes"):
        HELPER["prepare"](release_tree, "patch", date(2026, 10, 4))


def test_current_metadata_matches_runtime_version():
    import snnlab

    assert HELPER["check"](ROOT) == snnlab.__version__
