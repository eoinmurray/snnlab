"""The public Lang API needs no implementation-module imports."""

import subprocess
import sys
from typing import get_type_hints

import pytest

from snnlab import lang


def test_all_public_exports_resolve():
    names = lang.__all__
    assert len(names) == len(set(names))
    assert {"ops", "readouts", "training"}.isdisjoint(lang.__all__)
    for name in names:
        assert getattr(lang, name) is not None
        assert name in dir(lang)
    with pytest.raises(AttributeError, match="missing_export"):
        _ = lang.missing_export


def test_package_alias_and_exports_are_lazy_and_tensor_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import snnlab as snn\n"
            "assert 'snnlab.lang' not in sys.modules\n"
            "assert 'lang' in dir(snn)\n"
            "assert snn.lang.Network('example').name == 'example'\n"
            "assert snn.lang.AdamW(lr=1e-3).config == {'lr': 1e-3}\n"
            "for name in snn.lang.__all__: getattr(snn.lang, name)\n"
            "assert not any(name == 'snnlab.sim' or name.startswith('snnlab.sim.') "
            "for name in sys.modules)\n"
            "assert 'torch' not in sys.modules\n"
            "assert 'matplotlib' not in sys.modules\n"
            "assert 'snnlab.lang.core' not in sys.modules\n"
            "assert 'snnlab.lang.compiler' not in sys.modules\n",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    "name",
    [
        "Signal",
        "ParameterRef",
        "Quantity",
        "Readout",
        "TrainSpec",
        "ParameterGroup",
        "ValidationResult",
        "compile",
        "custom_operation",
    ],
)
def test_public_type_annotations_resolve(name):
    assert get_type_hints(getattr(lang, name))


def test_pinned_recipe_operations_alias_uses_the_same_implementations():
    assert lang.ops.reduce is lang.reduce
    assert lang.ops.custom is lang.custom_operation


def test_package_root_rejects_unknown_exports():
    import snnlab as snn

    with pytest.raises(AttributeError, match="missing_export"):
        _ = snn.missing_export
