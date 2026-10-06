"""Legacy CLI policy; removed switches retain their defaults for config replay."""

import warnings


class LegacyCLIWarning(FutureWarning):
    """Visible notice that a CLI argument belongs to deprecated legacy execution."""


REMOVED_LEGACY_FLAGS = frozenset(
    [
        "--adapt-strength-init-mv",
        "--adapt-strength-max-mv",
        "--adapt-tau-bounds-ms",
        "--adaptive-threshold",
        "--exact-k-initialization",
        "--independent-drive",
        "--independent-drive-i",
        "--lyapunov-eps",
        "--no-adaptive-threshold",
        "--no-dales-law",
        "--no-readout-bias",
        "--no-signed-readout",
        "--no-train-leak",
        "--quenched-drive",
        "--quenched-drive-i",
        "--readout-bias",
        "--readout-w-out-scale",
        "--recurrent-initial-zero-fraction",
        "--scale-w-ei",
        "--scale-w-ie",
        "--signed-readout",
        "--state-clamp",
        "--tau-m-e-bounds-ms",
        "--tau-m-i-bounds-ms",
        "--train-leak",
        "--trainable-w-ee",
        "--trainable-w-ii",
        "--transition-bundle",
        "--transition-end-ms",
        "--transition-start-ms",
        "--w-ee",
        "--w-ei",
        "--w-ie",
        "--w-ii",
    ]
)

DEPRECATED_LEGACY_FLAGS = frozenset(
    [
        "--model",
        "--n-hidden",
        "--n-in",
        "--n-inh",
        "--dales-law",
        "--ei-strength",
        "--ei-ratio",
        "--private-w-in",
        "--dt",
        "--tau-gaba",
        "--refractory-e-ms",
        "--refractory-i-ms",
        "--refractory-policy",
        "--readout",
        "--readout-w-init-mean",
        "--readout-w-init-std",
        "--w-in",
        "--w-ei-mean",
        "--w-ie-mean",
        "--w-in-initial-zero-fraction",
        "--surrogate-slope",
        "--trainable-w-ei",
        "--trainable-w-ie",
        "--input",
        "--dataset",
        "--digit",
        "--sample",
        "--sample-index",
        "--outputs",
        "--output-fields",
        "--recording-start-step",
        "--recording-mode",
        "--skip-load",
        "--perturb-mode",
        "--perturb-level",
        "--i-override-file",
        "--scale-w-in",
        "--lr",
        "--weight-decay",
        "--v-grad-dampen",
        "--fr-reg-upper-target-hz",
        "--fr-reg-upper-strength",
        "--wipe-dir",
        "--infer",
        "--load-config",
    ]
)

LEGACY_CONFIG_DEFAULTS = {
    "signed_readout": False,
    "readout_bias": False,
    "state_clamp": False,
    "train_leak": False,
    "tau_m_e_bounds_ms": None,
    "tau_m_i_bounds_ms": None,
    "adaptive_threshold": False,
    "adapt_tau_bounds_ms": None,
    "adapt_strength_init_mv": 1.0,
    "adapt_strength_max_mv": None,
    "recurrent_initial_zero_fraction": 0.0,
    "independent_drive": None,
    "independent_drive_i": None,
    "quenched_drive": None,
    "quenched_drive_i": None,
    "exact_k_initialization": False,
    "lyapunov_eps": 0.0,
    "readout_w_out_scale": 1.0,
    "w_ei": None,
    "w_ie": None,
    "w_ii": None,
    "w_ee": None,
    "trainable_w_ee": False,
    "trainable_w_ii": False,
    "scale_w_ei": 1.0,
    "scale_w_ie": 1.0,
    "transition_bundle": None,
    "transition_start_ms": None,
    "transition_end_ms": None,
}


# Only parent-parser fields were historically restored by --load-config.
LEGACY_REPLAY_FIELDS = frozenset(LEGACY_CONFIG_DEFAULTS) - {
    "scale_w_ei",
    "scale_w_ie",
    "transition_bundle",
    "transition_start_ms",
    "transition_end_ms",
}


def mark_legacy_help(parser):
    """Mark retained legacy switches without warning on help or implicit defaults."""
    for action in parser._actions:
        if DEPRECATED_LEGACY_FLAGS.intersection(action.option_strings) and not (
            action.help or ""
        ).startswith("[deprecated legacy]"):
            action.help = "[deprecated legacy] " + (action.help or "")


def warn_legacy_flags(argv):
    explicit = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    deprecated = sorted(explicit & DEPRECATED_LEGACY_FLAGS)
    if deprecated:
        warnings.warn(
            "Deprecated legacy CLI arguments: "
            + ", ".join(deprecated)
            + ". Author networks with snnlab.lang and use --executor graph --bundle. "
            "These arguments remain supported for older experiments.",
            LegacyCLIWarning,
            stacklevel=3,
        )
