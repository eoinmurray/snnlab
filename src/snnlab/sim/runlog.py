"""pinglab-cli run logging — one event model, three renderings.

Every semantic moment of a run (its config, each setup phase, each training
epoch, a warning, the final summary) is emitted once through this module and
rendered three ways:

  • terminal (a TTY)  — ANSI colour, aligned columns; an instrument-panel look
  • output.log        — the same lines, ANSI stripped (greppable plain text)
  • run.jsonl         — one typed JSON object per event; the canonical machine
                        record. Read THIS for deterministic parsing: it is
                        lossless and self-contained (config first, summary last).

The terminal is for the human watching a run; run.jsonl is for the notebook or
agent reading it back. output.log is the middle ground — a plain mirror of the
terminal. Colour never reaches output.log or run.jsonl.

Visual language
    ◆ banner   the run identity (mode · model · what drives it)
    ▸ phase    a setup step (loading / building / compiling) + its cost
    a stream   the per-epoch table, box-ruled into acc | dynamics | timing
    ✓ ⚠ ✗      success / caution / failure, coloured green / yellow / red
    → output   where the artifacts landed

Stable public names (imported by cli.py, train.py, and test_run_log.py):
    c, bold, dim, cyan, green, yellow, red, _strip_ansi, WIDTH
    MetricsJsonl, write_test_predictions, EventLog
    init_events, event, close_events
    WarningTracker, Heartbeat, format_bytes, format_eta, list_output_files
    banner, config_block, phase, epoch_header, epoch_row, metrics_line,
        summary, done, print_intro, print_progress_header, print_epoch,
        print_summary
    _env_hash, _git_sha, provenance, run_id
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ── ANSI palette ─────────────────────────────────────────────────────────

WIDTH = 70  # inner width of banners, rules, and the epoch stream

_IS_TTY = sys.stdout.isatty()


def c(code: str, text: str) -> str:
    """Wrap text in an ANSI SGR code — only when stdout is a TTY."""
    if not _IS_TTY:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def bold(text):
    return c("1", text)


def dim(text):
    return c("2", text)


def cyan(text):
    return c("36", text)


def green(text):
    return c("32", text)


def yellow(text):
    return c("33", text)


def red(text):
    return c("31", text)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _vis_len(s: str) -> int:
    """Visible length — ANSI codes are zero-width."""
    return len(_strip_ansi(s))


# Glyphs — kept in one place so the whole look shifts from here.
_BANNER = "◆"
_RULE = "─"
_RULE_HEAVY = "━"
_BAR = "│"
_PHASE = "▸"
_OK = "✓"
_WARN = "⚠"
_ERR = "✗"
_ARROW = "→"
_BEST = "↑"
_CLOCK = "◷"
_STATE = "◈"
_DOT = "·"


def _rule(heavy: bool = False) -> str:
    return dim((_RULE_HEAVY if heavy else _RULE) * WIDTH)


# ── Terminal cursor ──────────────────────────────────────────────────────
# During the in-place epoch progress the cursor would otherwise sit stranded
# mid-table. Hide it while progress is live and restore it when the run ends;
# an atexit backstop guarantees the terminal is left usable even on a crash.

_cursor_hidden = False


def _hide_cursor() -> None:
    global _cursor_hidden
    if _IS_TTY and not _cursor_hidden:
        import atexit

        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        _cursor_hidden = True
        atexit.register(_show_cursor)


def _show_cursor() -> None:
    global _cursor_hidden
    if _IS_TTY and _cursor_hidden:
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()
        _cursor_hidden = False


# ── run.jsonl event spine ────────────────────────────────────────────────


class EventLog:
    """Append-only JSONL writer: one typed object per semantic event.

    This is the canonical machine record of a run. Each line is
    {"ts": <iso>, "event": <type>, ...fields}. Types, in order of a run:
    run_start, config, phase, epoch, event, warning, summary. A reader can
    replay the whole run from this file alone.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def emit(self, event: str, **fields):
        rec = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        self._f.write(json.dumps(rec, default=float) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


# Module-global sink so the render helpers can mirror to run.jsonl without
# every caller threading an EventLog through. init_events() opens it;
# event() is a no-op until then, so importing runlog outside a run is safe.
_EVENTS: EventLog | None = None


def init_events(out_dir) -> None:
    """Open run.jsonl under out_dir and make event() live."""
    global _EVENTS
    if _EVENTS is not None:  # close a prior run's file rather than leak its handle
        _EVENTS.close()
    _EVENTS = EventLog(Path(out_dir) / "run.jsonl")


def event(event_type: str, **fields) -> None:
    """Emit a structured event to run.jsonl (no-op if not initialised)."""
    if _EVENTS is not None:
        _EVENTS.emit(event_type, **fields)


def close_events() -> None:
    global _EVENTS
    if _EVENTS is not None:
        _EVENTS.close()
        _EVENTS = None


# ── Metrics sidecar (per-epoch timeseries; consumed by notebooks) ─────────


class MetricsJsonl:
    """Append-only JSONL writer for per-epoch (or per-step) metrics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "w")

    def write(self, **fields):
        fields.setdefault(
            "timestamp", datetime.datetime.now().isoformat(timespec="seconds")
        )
        self._f.write(json.dumps(fields) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def write_test_predictions(path: Path, predictions: list):
    """Save list of {idx, true, pred, correct, logits} records to JSON."""
    with open(path, "w") as f:
        json.dump(predictions, f, indent=2, default=float)


# ── Provenance ───────────────────────────────────────────────────────────


def _git_sha() -> str:
    """Return current git SHA with '(dirty)' suffix if uncommitted changes.

    Honors PINGLAB_GIT_SHA env var as a fallback so remote containers (which
    don't have .git mounted) still record the host's SHA.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
        dirty = subprocess.call(
            ["git", "diff-index", "--quiet", "HEAD"],
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            timeout=2,
        )
        return f"{sha} (dirty)" if dirty else sha
    except Exception:
        return os.environ.get("PINGLAB_GIT_SHA", "unknown")


def _env_hash() -> str:
    """Hash of uv.lock (or pyproject.toml fallback) for env reproducibility."""
    for name in ("uv.lock", "pyproject.toml"):
        p = Path(name)
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            return f"{name}:{h}"
    return "unknown"


def run_id() -> str:
    """Compact run ID: r-YYYYMMDD-HHMMSS."""
    now = datetime.datetime.now()
    return f"r-{now.strftime('%Y%m%d-%H%M%S')}"


def provenance() -> dict:
    """Return a provenance dict to embed in config.json.

    `device` here is the best accelerator *available* on the host; the human
    log reports the device a run actually *used* at its done/summary line, so
    the two can differ (e.g. a CPU-only sim on an MPS Mac) without either
    being wrong.
    """
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return {
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "device": device,
        "python_env_hash": _env_hash(),
        "run_id": run_id(),
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


# ── Formatters ───────────────────────────────────────────────────────────


def format_eta(seconds: float) -> str:
    """Compact ETA string: 8m30s, 45s, 1h12m."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def format_bytes(n: int) -> str:
    size: float = n
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


# ── Banner + config block ────────────────────────────────────────────────


def banner(log, mode: str, model: str, subtitle: str) -> None:
    """Top-of-run identity line: ◆ pinglab · <mode> · <model>   <subtitle>.

    subtitle is the drive descriptor ("on mnist" when a dataset feeds the net,
    "synthetic-spikes" when it is driven synthetically) — the header never
    claims a dataset the run does not read.
    """
    left = f"{cyan(_BANNER)} {bold('pinglab')} {dim(_DOT)} {mode} {dim(_DOT)} {cyan(model)}"
    pad = max(1, WIDTH - _vis_len(left) - _vis_len(subtitle))
    log.info(left + " " * pad + dim(subtitle))
    log.info(_rule(heavy=True))
    event("run_start", mode=mode, model=model, subtitle=subtitle)


def config_block(log, groups: list) -> None:
    """Print curated config as aligned label/value rows.

    groups: list of (label, value) — value is a pre-composed one-liner, e.g.
        ("network", "784 → 1024 exc · 256 inh → 10 · 1.3M params").
    The exhaustive config lives in config.json / run.jsonl; this is the
    human-legible headline, not a dump.
    """
    fields = {}
    for label, value in groups:
        if value in (None, ""):
            continue
        log.info(f"  {dim(f'{label:<9}')} {value}")
        fields[label] = _strip_ansi(str(value))
    log.info(_rule())
    event("config", fields=fields)


# ── Setup phases ─────────────────────────────────────────────────────────


def warn(log, msg: str) -> None:
    """A standalone caution line: ⚠ <msg> (yellow), mirrored to run.jsonl."""
    log.info(f"  {yellow(_WARN)} {yellow(msg)}")
    event("warning", kind="notice", message=_strip_ansi(msg))


def phase(log, name: str, detail: str = "", elapsed_s: float | None = None) -> None:
    """One setup step: ▸ <name> ............... <detail | time>.

    Right-hand column is the cost (a time, a param count) so the eye can scan
    the setup ledger. Announced *before* the slow work when detail is unknown;
    call again after with the elapsed time to close it out if desired.
    """
    right = detail
    if elapsed_s is not None:
        t = format_eta(elapsed_s) if elapsed_s >= 1 else f"{elapsed_s:.1f}s"
        right = f"{detail}  {t}" if detail else t
    left = f"  {cyan(_PHASE)} {name}"
    pad = max(1, WIDTH - _vis_len(left) - _vis_len(right))
    log.info(left + " " * pad + dim(right))
    event("phase", name=name, detail=detail, elapsed_s=elapsed_s)


# ── Per-epoch progress stream ────────────────────────────────────────────
#
# Header, rule, and every row are built from ONE width spec (_EPOCH_FMT) so
# the three lines always align. Colour is applied to already-padded cells, so
# the zero-width ANSI codes never disturb the columns.


# ONE column spec drives the header, the underline, and every row — so they
# physically cannot drift apart. (key, header, width). Two rules hold columns
# aligned in EVERY terminal:
#   1. cells contain ASCII + digits only — no box-drawing, Greek, em-dash or
#      subscripts, whose width terminals disagree on (East-Asian "ambiguous").
#   2. units live in the header, cells are bare numbers, and widths are wide
#      enough that a value never overflows and shoves the row sideways.
# The column-group separator is itself a (key, header, width) tuple so the whole
# list is uniformly typed — the render helpers can index col[2] and do width
# arithmetic without losing the int type. Separators are found by identity
# (col is _SEP), never by value, so the empty key/zero width never collide with a
# real column.
_SEP: tuple[str, str, int] = ("", "", 0)
_EPOCH_COLS: list[tuple[str, str, int]] = [
    ("ep", "epoch", 9),   # "1000/1000"
    ("acc", "acc%", 5),
    ("loss", "loss", 7),
    _SEP,
    ("E", "E/Hz", 5),
    ("I", "I/Hz", 5),
    ("cv", "CV", 5),
    ("act", "act%", 5),
    _SEP,
    ("dt", "dt", 5),      # elapsed, a human duration (self-unit: 4s)
    ("eta", "eta", 7),
]


def _epoch_line(vals: dict, *, header: bool = False, best: bool = False) -> str:
    """Render one line from the shared column spec.

    header=True → the whole line is dimmed as the column titles; otherwise it is
    a data row (the │ separators are dimmed individually, and the accuracy cell
    is greened when best=True). Padding is computed on the plain value, then
    colour (zero visible width) is applied, so columns never shift.
    """
    parts = []
    for col in _EPOCH_COLS:
        if col is _SEP:
            parts.append(_BAR if header else dim(_BAR))
            continue
        key, _, w = col
        cell = f"{str(vals.get(key, '')):>{w}}"
        if best and key == "acc":
            cell = green(cell)
        parts.append(cell)
    line = "  " + "  ".join(parts)
    return dim(line) if header else line


def epoch_header(log) -> None:
    """Column titles + a rule spanning exactly the header's visible width."""
    labels = {key: hdr for col in _EPOCH_COLS if col is not _SEP
              for (key, hdr, _) in [col]}
    head = _epoch_line(labels, header=True)
    log.info(head)
    log.info(dim("  " + _RULE * (len(_strip_ansi(head)) - 2)))


# Backwards-compatible alias (train.py historically called this name).
def print_progress_header(log) -> None:
    epoch_header(log)


def epoch_row(
    log,
    ep: int,
    total: int,
    acc: float,
    loss: float,
    e_rate: float,
    i_rate: float | None,
    cv: float,
    activity: float,
    elapsed_s: float,
    eta_s: float,
    new_best: bool = False,
    warnings: list | None = None,
) -> None:
    """One epoch row + its run.jsonl event. Cells are bare (units in header)."""
    vals = {
        "ep": f"{ep}/{total}",
        "acc": f"{acc:.0f}",
        "loss": f"{loss:.3f}",
        "E": f"{e_rate:.0f}",
        "I": f"{i_rate:.0f}" if i_rate is not None else "-",
        "cv": f"{cv:.2f}",
        "act": f"{activity:.0f}",
        "dt": f"{int(elapsed_s)}s",
        "eta": format_eta(eta_s),
    }
    line = _epoch_line(vals, best=new_best)
    if new_best:
        line += " " + green(_BEST)
    if warnings:
        line += " " + " ".join(warnings)
    log.info(line)
    event(
        "epoch",
        ep=ep,
        total=total,
        acc=acc,
        loss=loss,
        rate_e=e_rate,
        rate_i=i_rate,
        cv=cv,
        act=activity,
        elapsed_s=elapsed_s,
        eta_s=eta_s,
        new_best=new_best,
        warnings=[_strip_ansi(w) for w in (warnings or [])],
    )


# Backwards-compatible alias (train.py historically called this name).
def print_epoch(*args, **kwargs) -> None:
    epoch_row(*args, **kwargs)


def epoch_progress(ep: int, total: int, note: str, elapsed_s: float,
                   loss: float | None = None) -> str:
    """A partial epoch-stream row for the epoch still in progress.

    Same column grid as a finished row, so everything lines up: the running
    `loss` sits under the loss column, the `note` (batch/eval counter) fills the
    as-yet-unknown dynamics columns between the bars, and elapsed sits under dt.
    Returned plain (bars uncoloured); Heartbeat dims the whole line.
    """
    w = {c[0]: c[2] for c in _EPOCH_COLS if c is not _SEP}
    ep_c = f"{f'{ep}/{total}':>{w['ep']}}"
    acc_c = " " * w["acc"]
    loss_c = f"{loss:>{w['loss']}.3f}" if loss is not None else " " * w["loss"]
    dt_c = f"{f'{int(elapsed_s)}s':>{w['dt']}}"
    eta_c = " " * w["eta"]
    mid_w = w["E"] + 2 + w["I"] + 2 + w["cv"] + 2 + w["act"]  # dynamics-group span
    parts = [ep_c, acc_c, loss_c, _BAR, f"{note:<{mid_w}}", _BAR, dt_c, eta_c]
    return "  " + "  ".join(parts)


# ── One-shot metrics (sim / probe) ───────────────────────────────────────


def metrics_line(log, m: dict, label: str = "result") -> None:
    """Render a firing-rate metrics dict as one dotted, glyph-led line.

    Human: ◈ result   E 16Hz · I 1Hz · CV 0.65 · act 84% · f₀ 10Hz
    Machine: an event with the raw numeric fields.
    """
    parts = [f"E {m.get('rate_e', 0):.0f}Hz", f"I {m.get('rate_i', 0):.0f}Hz",
             f"CV {m.get('cv', 0):.2f}", f"act {m.get('act', 0):.0%}"]
    if m.get("f0"):
        parts.append(f"f0 {m['f0']:.0f}Hz")
    body = f" {dim(_DOT)} ".join(parts)
    log.info(f"  {cyan(_STATE)} {dim(f'{label:<9}')} {body}")
    event("metrics", label=label, **{k: m.get(k) for k in
          ("rate_e", "rate_i", "cv", "act", "f0")})


# ── Summary block ────────────────────────────────────────────────────────


def list_output_files(out_dir: Path) -> list:
    """List all files in out_dir (recursively) with sizes."""
    out_dir = Path(out_dir)
    files = []
    if not out_dir.exists():
        return files
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out_dir)
            files.append((str(rel), p.stat().st_size))
    return files


def _summary_row(log, glyph, glyph_color, label, value) -> None:
    log.info(f"  {glyph_color(glyph)} {dim(f'{label:<9}')} {value}")


def summary(
    log,
    *,
    best_acc: float | None = None,
    final_acc: float | None = None,
    best_epoch: int | None = None,
    total_epochs: int | None = None,
    runtime_s: float | None = None,
    perf: dict | None = None,
    device: str | None = None,
    dynamics: dict | None = None,
    out_dir: Path | None = None,
    warnings: list | None = None,
) -> None:
    """The closing block: result, timing/cost, dynamics, warnings, artifacts."""
    log.info(_rule())

    if best_acc is not None:
        ep_str = ""
        if best_epoch:
            ep_str = dim(f"epoch {best_epoch}" + (f"/{total_epochs}" if total_epochs else ""))
        val = bold(green(f"{best_acc:.1f}%"))
        if final_acc is not None and final_acc != best_acc:
            val += f"   {ep_str}   {dim('final')} {final_acc:.1f}%"
        elif ep_str:
            val += f"   {ep_str}"
        _summary_row(log, _OK, green, "best", val)

    # Timing + cost on one line — the numbers the repo cares about (throughput,
    # peak memory) that were previously buried in metrics.json.
    if runtime_s is not None:
        bits = [format_eta(runtime_s)]
        if perf:
            if perf.get("epoch_warm_mean_s"):
                bits.append(f"{perf['epoch_warm_mean_s']:.1f}s/epoch")
            if perf.get("samples_per_sec_warm"):
                bits.append(f"{perf['samples_per_sec_warm']:.0f} samp/s")
            if perf.get("peak_memory_bytes"):
                bits.append(f"{format_bytes(perf['peak_memory_bytes'])} peak")
        if device:
            bits.append(device)
        _summary_row(log, _CLOCK, dim, "runtime",
                     f" {dim(_DOT)} ".join(bits))

    if warnings:
        _summary_row(log, _WARN, yellow,
                     "dynamics", yellow(" · ".join(_strip_ansi(w).strip("⚠ ")
                                                    for w in warnings)))

    if dynamics:
        body = f" {dim(_DOT)} ".join(
            f"{k} {v}" for k, v in dynamics.items() if v is not None
        )
        _summary_row(log, _STATE, cyan, "end", body)

    files = list_output_files(out_dir) if out_dir is not None else []
    if files:
        assert out_dir is not None  # files is non-empty only when out_dir is set
        _summary_row(log, _ARROW, cyan, "output",
                     cyan(str(Path(out_dir).resolve()) + "/"))
        # Collapse subdirectories to one line; list top-level files with sizes.
        dir_groups: dict = {}
        standalone = []
        for rel, sz in files:
            parts = rel.split(os.sep)
            if len(parts) > 1:
                dir_groups.setdefault(parts[0], []).append(sz)
            else:
                standalone.append((rel, sz))
        chips = [f"{rel} {dim(format_bytes(sz))}" for rel, sz in standalone]
        for d, sizes in dir_groups.items():
            chips.append(f"{d}/ {dim(f'({len(sizes)} files)')}")
        # wrap chips under the path, indented
        line = "            "
        for chip in chips:
            add = ("" if line.strip() == "" else f" {dim(_DOT)} ") + chip
            if _vis_len(line) + _vis_len(add) > WIDTH:
                log.info(line)
                line = "            " + chip
            else:
                line += add
        if line.strip():
            log.info(line)

    log.info(_rule())
    event(
        "summary",
        best_acc=best_acc,
        final_acc=final_acc,
        best_epoch=best_epoch,
        runtime_s=runtime_s,
        perf=perf,
        device=device,
        dynamics=dynamics,
        warnings=[_strip_ansi(w) for w in (warnings or [])],
    )


def done(log, elapsed_s: float, device: str | None = None) -> None:
    """Final line: ✓ done  <elapsed> · <device>."""
    _show_cursor()  # progress is over — hand the cursor back
    bits = [format_eta(elapsed_s)]
    if device:
        bits.append(device)
    _summary_row(log, _OK, green, "done", f" {dim(_DOT)} ".join(bits))
    event("done", elapsed_s=elapsed_s, device=device)


# ── Within-loop heartbeat ────────────────────────────────────────────────


class Heartbeat:
    """Within-epoch progress for a long batch/eval loop.

    A full-MNIST epoch is ~875 batches over minutes; without feedback the
    stream looks hung. Two renderings, chosen by whether stdout is a TTY:

      • terminal (TTY) — rewrites ONE line in place (carriage return), so a
        live counter climbs where the finished row will land, instead of a
        wall of lines. Call clear() before the finished row prints so the row
        overwrites the live line cleanly.
      • non-TTY (output.log, remote pod) — emits a plain line every `log_interval`
        s. A periodic record with no control characters; clear() is a no-op.

    `beat()` takes an already-formatted line (see epoch_progress) and dims it.
    """

    def __init__(self, tty_interval: float = 0.25, log_interval: float = 5.0):
        self._tty = _IS_TTY
        self._interval = tty_interval if self._tty else log_interval
        self._last = time.perf_counter()
        self._pending = False  # a live line is on screen, awaiting clear()

    def beat(self, log, line: str):
        now = time.perf_counter()
        if now - self._last < self._interval:
            return
        self._last = now
        line = line.rstrip()
        if self._tty:
            _hide_cursor()  # keep the cursor out of the live table
            sys.stdout.write("\r\x1b[2K" + dim(line))  # \r + clear-to-EOL
            sys.stdout.flush()
            self._pending = True
        else:
            log.info(dim(line))

    def clear(self):
        """Erase the pending live line (TTY only) so the next full log line
        prints cleanly in its place. No-op in non-TTY or if nothing pending."""
        if self._tty and self._pending:
            sys.stdout.write("\r\x1b[2K")
            sys.stdout.flush()
            self._pending = False


# ── Warning detector ─────────────────────────────────────────────────────


class WarningTracker:
    """Tracks rolling dynamics state to flag dead / saturated / no-progress."""

    def __init__(self):
        self.dead_streak = 0
        self.saturated_streak = 0
        self.best_acc = 0.0
        self.best_epoch = 0
        self.no_progress_since = 0
        self.observed_warnings = []  # (ep_start, ep_end, kind) aggregated

    def tick(
        self,
        ep: int,
        acc: float,
        activity: float,
        loss: float | None = None,
        grad_clip_frac: float = 0.0,
    ):
        flags = []
        # Activity flags only trigger when paired with no improvement —
        # extreme firing rates alone aren't pathological if the network
        # is still learning. Sparse-coding nets can run at ~1% activity;
        # gamma-locked PING can sustain >80% — both fine if accuracy climbs.
        stuck = self.no_progress_since >= 5

        # dead: silent for 3+ epochs AND not improving
        if activity < 1.0:
            self.dead_streak += 1
            if self.dead_streak >= 3 and stuck:
                flags.append(yellow(f"{_WARN} dead"))
                self._record(ep, "dead")
        else:
            self.dead_streak = 0

        # saturated: nearly-all-firing for 3+ epochs AND not improving
        if activity > 95.0:
            self.saturated_streak += 1
            if self.saturated_streak >= 3 and stuck:
                flags.append(yellow(f"{_WARN} saturated"))
                self._record(ep, "saturated")
        else:
            self.saturated_streak = 0

        # NaN
        if loss is not None and (loss != loss):  # NaN check
            flags.append(red(f"{_WARN} NaN"))
            self._record(ep, "NaN")

        # grad explosion
        if grad_clip_frac > 0.5:
            flags.append(yellow(f"{_WARN} grad-clip"))
            self._record(ep, "grad-clip")

        # new best tracking
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_epoch = ep
            self.no_progress_since = 0
        else:
            self.no_progress_since += 1

        if self.no_progress_since >= 10:
            flags.append(yellow(f"{_WARN} stuck"))
            self._record(ep, "stuck")

        return flags

    def _record(self, ep: int, kind: str):
        # Extend existing run of same kind or start new
        if (
            self.observed_warnings
            and self.observed_warnings[-1][2] == kind
            and self.observed_warnings[-1][1] == ep - 1
        ):
            s, _, k = self.observed_warnings[-1]
            self.observed_warnings[-1] = (s, ep, k)
        else:
            self.observed_warnings.append((ep, ep, kind))

    def summary_lines(self) -> list:
        """Human-readable aggregated warnings for the summary block."""
        if not self.observed_warnings:
            return []
        lines = []
        for start, end, kind in self.observed_warnings:
            span = f"ep {start}" if start == end else f"ep {start}–{end}"
            lines.append(f"{kind} {span}")
        return lines


# ── Legacy shims (print_intro / print_summary) ───────────────────────────
#
# The pre-redesign entry points. cli.py / train.py now call banner+config and
# summary() directly; these remain so any stray caller keeps working.


def print_intro(log, mode: str, model: str, subtitle: str, sections: dict):
    banner(log, mode, model, subtitle)
    groups = []
    for section_name, items in sections.items():
        if not items:
            continue
        vals = " · ".join(f"{k} {v}" for k, v in items.items() if v not in (None, ""))
        groups.append((section_name.lower(), vals))
    config_block(log, groups)


def print_summary(log, **kwargs):
    summary(log, **kwargs)
