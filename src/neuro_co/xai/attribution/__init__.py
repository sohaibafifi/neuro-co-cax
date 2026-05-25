"""Attribution methods for autoregressive CO policies.

- `AttributionTrace`, common dataclass returned by every method.
- `gradient_attribution`, gradient x feature (cheap baseline).
- `contrastive_attribution`, gradient of `log pi(a) - log pi(b)`.
- `integrated_gradients`, Riemann-sum path integral.
- `deeplift_attribution`, DeepLIFT-Rescale via backward hooks.

Method-specific modules live alongside this file; `_common` holds the
shared helpers (feature collection, rollout driver, trace packing).
"""

from __future__ import annotations

from neuro_co.xai.attribution._common import AttributionTrace, drop_baseline_keys, step_logits
from neuro_co.xai.attribution.contrastive import contrastive_attribution
from neuro_co.xai.attribution.deeplift import deeplift_attribution
from neuro_co.xai.attribution.gradient import gradient_attribution
from neuro_co.xai.attribution.ig import integrated_gradients

__all__ = [
    "AttributionTrace",
    "contrastive_attribution",
    "deeplift_attribution",
    "drop_baseline_keys",
    "gradient_attribution",
    "integrated_gradients",
    "step_logits",
]
