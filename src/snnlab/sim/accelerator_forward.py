"""Forward-only accelerator checks for the graph-native PING executor.

This module deliberately keeps the validation case small and explicit.  It
checks graph-native repeatability on one accelerator and numerical agreement
between the legacy and graph-native implementations on that accelerator.  It
does not cover training, checkpoints, or cross-device equality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch  # noqa: E402

from snnlab import lang as snn  # noqa: E402
from snnlab.sim import models as M  # noqa: E402
from snnlab.sim.conformance import (  # noqa: E402
    ComparisonPolicy,
    ConformanceReport,
    compare_conformance_layers,
)
from snnlab.sim.execution import (  # noqa: E402
    ExecutionSpec,
    GraphExecutor,
    build,
    export_legacy_parameters_v1,
    import_legacy_parameters_v1,
)

ATOL = 1e-6
RTOL = 1e-6


@dataclass(frozen=True)
class ForwardAcceleratorResult:
    device: str
    snnlang_reproducibility: ConformanceReport
    legacy_graph_parity: ConformanceReport

    @property
    def passed(self) -> bool:
        return self.snnlang_reproducibility.passed and self.legacy_graph_parity.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "passed": self.passed,
            "tolerances": {"atol": ATOL, "rtol": RTOL},
            "snnlang_reproducibility": self.snnlang_reproducibility.to_dict(),
            "legacy_graph_parity": self.legacy_graph_parity.to_dict(),
        }

    def require_passed(self) -> None:
        self.snnlang_reproducibility.require_passed()
        self.legacy_graph_parity.require_passed()


def _author_ping() -> snn.Bundle:
    net = snn.Network("accelerator_forward_ping", dt=0.1 * snn.ms)
    events = net.input(
        "events", shape=("time", "batch", 2), signal_type="spikes", unit="spike"
    )
    cell = snn.components.ping(
        net,
        name="cell",
        n_e=4,
        n_i=1,
        source=events,
        include_silent_recurrence=True,
    )
    scores = snn.readouts.MeanVoltage(source=cell.E.spikes, classes=2, name="scores")
    net.output("class_logits", scores)
    return snn.compile(net)


def _fixed_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    events = torch.zeros(40, 2, 2, device=device)
    events[:, 0, 0] = 1
    events[::2, 1, 1] = 1
    return {"events": events}


def _fixed_graph_model(bundle: snn.Bundle, device: torch.device) -> GraphExecutor:
    built = build(
        ExecutionSpec(kind="build", executor="graph", graph=bundle.graph, seed=7)
    )
    if not isinstance(built.model, GraphExecutor):
        raise TypeError("graph build did not return GraphExecutor")
    model = built.model
    parameters = model.parameter_map()
    with torch.no_grad():
        parameters["cell_input.weight"].fill_(10.0)
        parameters["cell_E_to_E.weight"].zero_()
        parameters["cell_E_to_I.weight"].fill_(2.0)
        parameters["cell_I_to_E.weight"].fill_(5.0)
        parameters["cell_I_to_I.weight"].zero_()
        parameters["scores_projection.weight"].copy_(
            torch.tensor([[1.0, 0.5], [0.25, 1.5], [1.25, 0.75], [0.5, 1.0]])
        )
    return model.to(device)


def _fixed_legacy_model(
    bundle: snn.Bundle, graph_model: GraphExecutor, device: torch.device
) -> M.COBANet:
    legacy = M.COBANet(
        hidden_sizes=[4],
        n_inh_per_layer={1: 1},
        readout_mode="mem-mean",
        w_in=(0.0, 0.0),
        w_hid=(0.0, 0.0),
        w_ee=(0.0, 0.0),
        w_ei=(0.0, 0.0),
        w_ie=(0.0, 0.0),
        w_ii=(0.0, 0.0),
    )
    legacy.recording = True
    exported = export_legacy_parameters_v1(bundle.graph, graph_model.parameter_map())
    imported = import_legacy_parameters_v1(bundle.graph, exported.parameters)
    with torch.no_grad():
        for name, value in imported.parameters.items():
            graph_model.parameter_map()[name].copy_(value)
        legacy_parameters = dict(legacy.named_parameters())
        for name, value in exported.parameters.items():
            legacy_parameters[name].copy_(value)
    return legacy.to(device)


def _graph_forward(
    model: GraphExecutor, inputs: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = model(inputs, record="full")
    return {
        "e_spikes": result.recordings["cell_E.spikes"],
        "i_spikes": result.recordings["cell_I.spikes"],
        "e_voltage": result.recordings["cell_E.voltage"],
        "i_voltage": result.recordings["cell_I.voltage"],
        "input_conductance": result.recordings["cell_input.conductance"],
        "e_to_i_conductance": result.recordings["cell_E_to_I.conductance"],
        "i_to_e_conductance": result.recordings["cell_I_to_E.conductance"],
        "logits": result.outputs["class_logits"],
    }


def _legacy_forward(
    model: M.COBANet, inputs: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    logits = model(input_spikes=inputs["events"])
    return {
        "e_spikes": model.spike_record["hid"],
        "i_spikes": model.spike_record["inh"],
        "e_voltage": model.spike_record["v_e_1"],
        "i_voltage": model.spike_record["v_i_1"],
        "input_conductance": model.spike_record["ge_e_1"],
        "e_to_i_conductance": model.spike_record["ge_i_1"],
        "i_to_e_conductance": model.spike_record["gi_e_1"],
        "logits": logits,
    }


def _cpu_copy(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().to(device="cpu") for name, value in values.items()}


def _run_worker(kind: str, device: str, output: Path) -> None:
    resolved = torch.device(device)
    M.N_IN = 2
    M.N_OUT = 2
    M.dt = 0.1
    M.T_ms = 4.0
    M.T_steps = 40
    bundle = _author_ping()
    inputs = _fixed_inputs(resolved)
    graph_model = _fixed_graph_model(bundle, resolved)
    if kind == "graph":
        values = _graph_forward(graph_model, inputs)
    elif kind == "legacy":
        legacy_model = _fixed_legacy_model(bundle, graph_model, resolved)
        values = _legacy_forward(legacy_model, inputs)
    else:
        raise ValueError(f"unknown forward accelerator worker {kind!r}")
    torch.save(_cpu_copy(values), output)


def _subprocess_forward(
    kind: str, device: str, output: Path
) -> dict[str, torch.Tensor]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "snnlab.sim.accelerator_forward",
            "--device",
            device,
            "--worker",
            kind,
            "--output",
            str(output),
        ],
        check=True,
    )
    loaded = torch.load(output, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in loaded.items()
    ):
        raise TypeError(f"invalid {kind} forward worker output")
    return loaded


def _policies(
    fields: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, ComparisonPolicy]]:
    return {
        "forward": {
            name: ComparisonPolicy(mode="numeric", atol=ATOL, rtol=RTOL)
            for name in fields
        }
    }


def run_forward_accelerator_check(device: str) -> ForwardAcceleratorResult:
    resolved = torch.device(device)
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if resolved.type not in {"mps", "cuda"}:
        raise ValueError(f"accelerator device expected mps or cuda, got {device!r}")

    with tempfile.TemporaryDirectory(prefix="snnlang-forward-") as scratch:
        root = Path(scratch)
        first = _subprocess_forward("graph", device, root / "graph-first.pt")
        second = _subprocess_forward("graph", device, root / "graph-second.pt")
        legacy = _subprocess_forward("legacy", device, root / "legacy.pt")
        reproducibility = compare_conformance_layers(
            "snnlang-same-accelerator-forward",
            {"forward": first},
            {"forward": second},
            policies=_policies(first),
        )
        parity = compare_conformance_layers(
            "legacy-snnlang-same-accelerator-forward",
            {"forward": legacy},
            {"forward": first},
            policies=_policies(legacy),
        )
    return ForwardAcceleratorResult(
        device=str(resolved),
        snnlang_reproducibility=reproducibility,
        legacy_graph_parity=parity,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check graph repeatability and legacy parity for one forward PING case."
    )
    parser.add_argument("--device", required=True, choices=("mps", "cuda"))
    parser.add_argument("--worker", choices=("graph", "legacy"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.output is None:
            parser.error("--worker requires --output")
        _run_worker(args.worker, args.device, args.output)
        return
    if args.output is not None:
        parser.error("--output requires --worker")
    result = run_forward_accelerator_check(args.device)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    result.require_passed()


if __name__ == "__main__":
    main()
