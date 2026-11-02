"""Write the Rust scorer's conformance fixtures from the pandas pipeline.

Each fixture directory is a plan bundle plus ``input.json`` (the request) and
``expected.json`` (what the pandas pipeline returns for it). ``tests/fixtures.rs``
scores every fixture and requires the same output, value for value.

Run from the repository root after changing the exporter or the test pipelines:

    python scorer/tests/make_fixtures.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cloudpickle  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.pyfunc import PythonModelContext  # noqa: E402

from open_rating_engine.scoring_plan import write_plan_bundle  # noqa: E402
from open_rating_engine.scoring_plan.parity import (  # noqa: E402
    _single_threaded_onnx,
    frames_to_records,
    pipeline_records,
)
from tests.scoring_plan_cases import kitchen_sink_inputs, kitchen_sink_pipeline, write_tiny_onnx  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EXAMPLES = ROOT / ".examples"
# The example's demand model is H2O, which the plan cannot score; drop it and the
# reporting layers that read its output.
H2O_AND_DEPENDENTS = [
    "risk.reporting_expected_profit_margin",
    "risk.reporting_expected_profit",
    "risk.reporting_expected_technical_premium",
    "risk.reporting_expected_gwp",
    "risk.conversion_rate",
]


def write_fixture(name: str, pipeline, inputs: dict) -> None:
    out = FIXTURES / name
    shutil.rmtree(out, ignore_errors=True)
    write_plan_bundle(pipeline, out)
    expected = pipeline_records(_single_threaded_onnx(pipeline), inputs)
    (out / "input.json").write_text(json.dumps({"inputs": inputs}, indent=1) + "\n", encoding="utf-8")
    (out / "expected.json").write_text(json.dumps({"predictions": expected}, indent=1) + "\n", encoding="utf-8")
    print(f"{name}: {sum(len(rows) for rows in expected.values())} output rows")


def kitchen_sink() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = kitchen_sink_pipeline(write_tiny_onnx(Path(tmp) / "tiny.onnx"))
        write_fixture("kitchen_sink", pipeline, kitchen_sink_inputs())


def motor_example(n_policies: int = 20) -> None:
    models = EXAMPLES / "models"
    pipeline = cloudpickle.load((models / "rating_engines" / "pricing_pipeline.pkl").open("rb"))
    for name in H2O_AND_DEPENDENTS:
        pipeline.remove_parameter(name)
    pipeline.context = PythonModelContext(
        artifacts={"motor_risk_cost_onnx": str(models / "motor_risk_cost_onnx" / "pipeline_onnx.onnx")},
        model_config={"environment": "local"},
    )
    pipeline.options["show_progress"] = False
    data_dir = EXAMPLES / "data" / "origination"
    frames = {t: pd.read_parquet(data_dir / f"{t}.parquet") for t in ["policy", "risk", "cover", "claims_history"]}
    chosen = set(frames["policy"]["policy.policy_number"].iloc[:n_policies])
    sample = {}
    for table, frame in frames.items():
        key = next(c for c in frame.columns if c.endswith("policy_number"))
        sample[table] = frame[frame[key].isin(chosen)].reset_index(drop=True)
    write_fixture("motor_example", pipeline, frames_to_records(sample))


if __name__ == "__main__":
    kitchen_sink()
    motor_example()
