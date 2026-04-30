from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pt_plot import PtFitResult


@dataclass
class AnalysisContext:
    pt_fit_result: PtFitResult | None = None
