"""Scoring plans: a pipeline compiled to JSON for the Rust scorer. Loaded lazily; see ``.._lazy``."""

from typing import TYPE_CHECKING

from .._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    exports={
        ".exporter": [
            "Blocker",
            "ExportResult",
            "PlanExportError",
            "export_plan",
            "write_plan_bundle",
        ],
        ".parity": [
            "ParityReport",
            "check_parity",
        ],
    },
)

if TYPE_CHECKING:
    from .exporter import (
        Blocker,
        ExportResult,
        PlanExportError,
        export_plan,
        write_plan_bundle,
    )
    from .parity import ParityReport, check_parity
