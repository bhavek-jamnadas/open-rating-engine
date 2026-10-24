"""open-rating-engine: a DAG-based insurance/banking rating engine.

Names below are loaded on first access (see ``_lazy.py``), so ``import
open_rating_engine`` is cheap and pulls in mlflow only once the pipeline or a layer
is actually used.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from ._lazy import attach

try:
    __version__ = version("open-rating-engine")
except PackageNotFoundError:  # running from a source checkout that isn't installed
    __version__ = "0+unknown"

__getattr__, __dir__, _exports = attach(
    __name__,
    exports={
        ".rating_engine.pricing_pipeline": ["PyfuncPricingPipeline"],
        ".rating_engine.pricing_layers": [
            "BasePricingLayer",
            "ColumnCalculation",
            "CustomLayer",
            "DropColumns",
            "GlmScoring",
            "H2OScoringParameter",
            "OnnxScoring",
            "RatingTablesLayer",
            "RenameColumns",
            "SourceColumn",
            "TableAggregation",
            "TableDrop",
            "TableJoin",
            "TableTransform",
            "DependencyError",
            "LayerCalculationError",
        ],
        ".data_processing.data_aging": ["apply_aging", "AgingRuleError"],
    },
    submodules=["data_processing", "optimization", "rating_engine"],
)
__all__ = ["__version__", *_exports]

if TYPE_CHECKING:
    from . import data_processing, optimization, rating_engine
    from .data_processing.data_aging import AgingRuleError, apply_aging
    from .rating_engine.pricing_layers import (
        BasePricingLayer,
        ColumnCalculation,
        CustomLayer,
        DependencyError,
        DropColumns,
        GlmScoring,
        H2OScoringParameter,
        LayerCalculationError,
        OnnxScoring,
        RatingTablesLayer,
        RenameColumns,
        SourceColumn,
        TableAggregation,
        TableDrop,
        TableJoin,
        TableTransform,
    )
    from .rating_engine.pricing_pipeline import PyfuncPricingPipeline
