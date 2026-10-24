"""Data preparation applied before scoring. Loaded lazily; see ``.._lazy``."""

from typing import TYPE_CHECKING

from .._lazy import attach

__getattr__, __dir__, __all__ = attach(
    __name__,
    exports={".data_aging": ["apply_aging", "AgingRuleError"]},
)

if TYPE_CHECKING:
    from .data_aging import AgingRuleError, apply_aging
