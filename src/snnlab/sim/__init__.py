"""Public simulation API, with implementations imported on demand.

Import supported functions and types from ``snnlab.sim``. The export map keeps
these names independent of the internal file layout.
"""

from importlib import import_module

from ._version import __version__

_EXPORTS = {
    "TrainingHistory": "training.history",
    "HistoryEpoch": "training.history",
    "MetricCurve": "training.history",
    "NetworkSnapshot": "artifacts.network_snapshot",
    "PhysicalParameter": "artifacts.network_snapshot",
    "RecordingData": "artifacts.recordings",
    "SignalInfo": "artifacts.recordings",
    "read_metrics": "artifacts.evidence",
    "DatasetPartition": "inputs.snapshots",
    "DatasetSnapshot": "inputs.snapshots",
    "stratified_split": "inputs.snapshots",
    "EpochLogger": "training.progress",
    "format_epoch": "training.progress",
    "Performance": "requests.performance",
    "load_dataset": "inputs.datasets",
    "encode_images_poisson": "inputs.encoders",
    "poisson_spikes": "graph.dynamics",
    "fast_sigmoid_spike": "graph.dynamics",
    "lif_step_expeuler": "graph.dynamics",
    "init_lif_state": "graph.dynamics",
    "ValidationSpec": "requests.spec",
    "TestSpec": "requests.spec",
    "InputBinding": "inputs.bindings",
    "DenseArrayBinding": "inputs.bindings",
    "TargetArrayBinding": "inputs.bindings",
    "EventStreamBinding": "inputs.bindings",
    "PoissonInputBinding": "inputs.bindings",
    "DatasetEncoder": "inputs.bindings",
    "DatasetSnapshotBinding": "inputs.bindings",
    "TrainingMetrics": "training.policies",
    "Batching": "training.policies",
    "EpochObservations": "training.observations",
    "ObservationProbe": "training.observations",
    "EpochReport": "training.policies",
    "build": "requests.api",
    "simulate": "requests.api",
    "train": "requests.api",
    "resolve_device": "requests.handlers",
    "GraphExecutor": "graph.executor",
    "GraphPlan": "graph.plan",
    "PlannedProjection": "graph.plan",
    "plan_graph": "graph.plan",
    "CapabilityIssue": "graph.plan",
    "graph_capability_issues": "graph.plan",
    "ExecutionResult": "requests.result",
    "NumpyExecutionResult": "requests.result",
    "TrainingCheckpoint": "training.checkpoints",
    "save_training_checkpoint": "training.checkpoints",
    "load_training_checkpoint": "training.checkpoints",
    "capture_training_rng_state": "training.checkpoints",
    "restore_training_rng_state": "training.checkpoints",
    "CheckpointSelection": "training.selection",
    "SelectionMetric": "training.selection",
    "GraphRuntimeState": "runtime.state",
    "save_runtime_state": "runtime.state",
    "load_runtime_state": "runtime.state",
    "runtime_state_compatibility": "runtime.state",
    "runtime_state_signature": "runtime.state",
    "SnapshotSpec": "artifacts.snapshots",
    "RunSnapshot": "artifacts.snapshots",
    "write_run_snapshot": "artifacts.snapshots",
    "write_inference_artifacts": "artifacts.inference",
    "validate_inference_artifacts": "artifacts.inference",
    "derive_inference_products": "artifacts.inference",
    "validate_derived_inference_products": "artifacts.inference",
    "retention_identity": "artifacts.inference",
    "MeasurementWindow": "runtime.recording",
    "SignalRecording": "runtime.recording",
    "RecordingSpec": "runtime.recording",
    "RecordingBlock": "runtime.recording",
    "NPZRecordingSink": "runtime.recording",
    "recording_identity": "runtime.recording",
    "ReplaySpikes": "runtime.interventions",
    "DropSpikes": "runtime.interventions",
    "AddPoissonSpikes": "runtime.interventions",
    "DenseSpikeReplay": "runtime.interventions",
    "SparseSpikeReplay": "runtime.interventions",
    "SpikeReplay": "runtime.interventions",
    "Intervention": "runtime.interventions",
    "save_spike_replay": "runtime.interventions",
    "load_spike_replay": "runtime.interventions",
    "intervention_identity": "runtime.interventions",
    "Boundary": "runtime.resets_decisions",
    "BoundarySchedule": "runtime.resets_decisions",
    "ResetVoltage": "runtime.resets_decisions",
    "DecisionSegments": "runtime.resets_decisions",
    "segment_identity": "runtime.resets_decisions",
    "TraceSpec": "diagnostics.tracing",
    "ResolvedDenseInputs": "inputs.bindings",
    "load_dense_array_bindings": "inputs.resolution",
    "load_target_array_bindings": "inputs.resolution",
    "load_event_stream_bindings": "inputs.resolution",
    "resolve_dense_array_bindings": "inputs.resolution",
    "resolve_event_stream_bindings": "inputs.resolution",
    "resolve_poisson_input_bindings": "inputs.resolution",
    "resolve_input_bindings": "inputs.resolution",
    "resolve_target_array_bindings": "inputs.resolution",
    "resolve_dataset_snapshot_binding": "inputs.resolution",
}

__all__ = [*_EXPORTS, "__version__"]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
