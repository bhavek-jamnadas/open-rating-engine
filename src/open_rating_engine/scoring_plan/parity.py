"""Check that the Rust scorer gives the same results as the pandas pipeline.

The check runs the pipeline's own ``predict`` on sample inputs, runs the Rust
scorer's ``score`` command on the same inputs, and compares every table, column
and row. Inputs travel as JSON records, exactly as a serving endpoint receives
them, so both sides see the same types.

A deploy should refuse the Rust runtime for a snapshot unless this check passes.
"""

import json
import math
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "Mismatch",
    "ParityReport",
    "check_parity",
    "compare_records",
    "frames_to_records",
    "pipeline_records",
    "run_rust_scorer",
]


@dataclass(frozen=True)
class Mismatch:
    """One difference between the pandas output and the Rust output."""

    table: str
    column: Optional[str]
    row: Optional[int]
    expected: object
    actual: object

    def __str__(self) -> str:
        where = self.table
        if self.column is not None:
            where += f".{self.column}"
        if self.row is not None:
            where += f"[{self.row}]"
        return f"{where}: pandas={self.expected!r} rust={self.actual!r}"


@dataclass
class ParityReport:
    """Result of :func:`check_parity`."""

    rows_checked: int
    values_checked: int
    mismatches: List[Mismatch] = field(default_factory=list)
    max_relative_error: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self, limit: int = 10) -> str:
        head = (
            f"{'PASS' if self.ok else 'FAIL'}: {self.values_checked} values in "
            f"{self.rows_checked} rows, largest relative difference {self.max_relative_error:.3g}"
        )
        if self.ok:
            return head
        lines = [head, f"{len(self.mismatches)} mismatches, first {min(limit, len(self.mismatches))}:"]
        lines += [f"  - {m}" for m in self.mismatches[:limit]]
        return "\n".join(lines)


def _json_safe(value):
    """The serving container's conversion: non-finite floats become null."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def frames_to_records(frames: Dict[str, pd.DataFrame]) -> Dict[str, list]:
    """``{table: DataFrame}`` as the JSON records a serving endpoint exchanges."""
    out = {}
    for name, frame in frames.items():
        records = frame.to_dict(orient="records")
        out[name] = json.loads(json.dumps(_json_safe(records), default=_json_default))
    return out


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def pipeline_records(pipeline, inputs: Dict[str, list]) -> Dict[str, list]:
    """Score JSON-record ``inputs`` with the pandas pipeline, the way the serving
    container does: records to DataFrames, ``predict``, records back."""
    frames = {name: pd.DataFrame(records) for name, records in inputs.items()}
    output = pipeline.predict(frames)
    # Outside the "local" environment predict returns arrays of record dicts.
    return frames_to_records({
        name: value if isinstance(value, pd.DataFrame) else pd.DataFrame(list(value))
        for name, value in output.items()
    })


def run_rust_scorer(
    command: Sequence[str],
    bundle: Union[str, Path],
    inputs: Dict[str, list],
    timeout: float = 600,
) -> Dict[str, list]:
    """Run the Rust scorer's ``score`` command and return its predictions.

    ``command`` is the program and any leading arguments, for example
    ``["ors-scorer"]`` or ``["docker", "run", "--rm", "-i", "-v", "{bundle}:/bundle",
    "ors-scorer"]``. ``{bundle}`` in any argument is replaced with the bundle path.
    ``score --bundle <path>`` is appended, using ``/bundle`` when the command
    mounts the bundle there.
    """
    bundle = str(Path(bundle).resolve())
    args = [a.replace("{bundle}", bundle) for a in command]
    in_container = any(":/bundle" in a for a in args)
    args += ["score", "--bundle", "/bundle" if in_container else bundle]
    proc = subprocess.run(
        args,
        input=json.dumps({"inputs": inputs}, allow_nan=False),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rust scorer failed (exit {proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    return json.loads(proc.stdout)["predictions"]


def _close(expected, actual, rtol: float, atol: float) -> tuple:
    """(equal, relative_error) for two JSON values."""
    if expected is None or actual is None:
        return expected is None and actual is None, 0.0
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual, 0.0
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if expected == actual:
            return True, 0.0
        diff = abs(float(expected) - float(actual))
        scale = max(abs(float(expected)), abs(float(actual)))
        rel = diff / scale if scale else math.inf
        return diff <= atol + rtol * scale, rel
    return expected == actual, 0.0


def compare_records(
    expected: Dict[str, list],
    actual: Dict[str, list],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> ParityReport:
    """Compare two ``{table: records}`` results value by value."""
    report = ParityReport(rows_checked=0, values_checked=0)
    if list(expected) != list(actual):
        report.mismatches.append(Mismatch("<tables>", None, None, list(expected), list(actual)))
    for table, exp_rows in expected.items():
        act_rows = actual.get(table)
        if act_rows is None:
            continue
        if len(exp_rows) != len(act_rows):
            report.mismatches.append(Mismatch(table, None, None, f"{len(exp_rows)} rows", f"{len(act_rows)} rows"))
            continue
        exp_cols = list(exp_rows[0]) if exp_rows else []
        act_cols = list(act_rows[0]) if act_rows else []
        if exp_cols != act_cols:
            report.mismatches.append(Mismatch(table, "<columns>", None, exp_cols, act_cols))
        for i, (exp_row, act_row) in enumerate(zip(exp_rows, act_rows)):
            report.rows_checked += 1
            for column, exp_value in exp_row.items():
                report.values_checked += 1
                if column not in act_row:
                    continue
                equal, rel = _close(exp_value, act_row[column], rtol, atol)
                if math.isfinite(rel):
                    report.max_relative_error = max(report.max_relative_error, rel)
                if not equal:
                    report.mismatches.append(Mismatch(table, column, i, exp_value, act_row[column]))
    return report


def check_parity(
    pipeline,
    bundle: Union[str, Path],
    samples: Union[Dict[str, pd.DataFrame], Dict[str, list]],
    command: Sequence[str],
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> ParityReport:
    """Score ``samples`` with the pandas pipeline and with the Rust scorer, and compare.

    Parameters
    ----------
    pipeline : PyfuncPricingPipeline
        The pipeline the bundle was exported from, with its context loaded.
    bundle : path
        Directory written by :func:`~.exporter.write_plan_bundle`.
    samples : dict
        ``{table: DataFrame}`` or ``{table: records}``.
    command : sequence of str
        How to run the scorer; see :func:`run_rust_scorer`.
    rtol, atol : float
        Allowed difference for numbers: ``|a - b| <= atol + rtol * max(|a|, |b|)``.

    Notes
    -----
    ONNX Runtime sums tree ensembles in an order that depends on its thread count,
    so a model's float32 outputs change with the number of cores once a batch is
    large. The Rust scorer runs ONNX on one thread by default; this check scores the
    pandas side on one thread too, on a copy of the pipeline, so the comparison is
    exact.
    """
    inputs = {
        name: frames_to_records({name: value})[name] if isinstance(value, pd.DataFrame) else value
        for name, value in samples.items()
    }
    expected = pipeline_records(_single_threaded_onnx(pipeline), inputs)
    actual = run_rust_scorer(command, bundle, inputs)
    return compare_records(expected, actual, rtol=rtol, atol=atol)


def _single_threaded_onnx(pipeline):
    """A copy of ``pipeline`` whose ONNX layers score on one thread."""
    from ..rating_engine.pricing_layers import OnnxScoring

    onnx_layers = [name for name, layer in pipeline.layers.items() if type(layer) is OnnxScoring]
    if not onnx_layers:
        return pipeline
    import onnxruntime as ort

    copy = deepcopy(pipeline)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for name in onnx_layers:
        layer = copy.layers[name]
        path = copy.context.artifacts[layer.model_artifact_key]
        layer.session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    return copy
