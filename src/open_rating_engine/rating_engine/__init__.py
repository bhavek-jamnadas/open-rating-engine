"""The pricing pipeline and its layer types. Loaded lazily; see ``.._lazy``."""

from typing import TYPE_CHECKING

from .._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    exports={
        ".pricing_pipeline": ["PyfuncPricingPipeline", "download_blob_to_temp"],
        ".pricing_layers": [
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
    },
)

if TYPE_CHECKING:
    from .pricing_layers import (
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
    from .pricing_pipeline import PyfuncPricingPipeline, download_blob_to_temp
