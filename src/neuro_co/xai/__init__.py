"""Action-level attribution + faithfulness for neural CO solvers.

The light half of the former `neuro-co-xai`: depends only on
`torch`, `tensordict`, and `numpy`. Encoder probing /
matplotlib-backed visualisation moved to `neuro-co-probe`.

Public surface
--------------

Attribution (per-step `[B, T, N]` gradient-x-feature traces):

- `gradient_attribution(...)` -- |grad x feature| of `log pi(a_t)`
- `contrastive_attribution(...)` -- |grad x feature| of margin
  `log pi(a) - log pi(b)`
- `deeplift_attribution(...)` -- DeepLIFT-Rescale via backward hooks
- `integrated_gradients(...)` -- midpoint Riemann IG path

Faithfulness:

- `deletion_flip_rate(...)` -- mask top-k features, count flips
- `sufficiency_keep_rate(...)` -- keep only top-k, count unchanged
- `sanity_check(...)` -- model-weight randomisation (Adebayo et al.)

Concept registry:

- `ConceptBank`, `register_concept_bank`, `concept_registry`,
  `infer_problem_name`

Cross-seed stability:

- `top_k_stability`, `load_explanation_traces`, `StabilityReport`
"""

from __future__ import annotations

from neuro_co.xai.attribution import (
    AttributionTrace,
    contrastive_attribution,
    deeplift_attribution,
    gradient_attribution,
    integrated_gradients,
)
from neuro_co.xai.concept import (
    ConceptBank,
    ConceptFn,
    InstanceRegistry,
    concept_registry,
    infer_problem_name,
    register_concept_bank,
)
from neuro_co.xai.faithfulness import (
    DeletionReport,
    SanityCheckReport,
    SufficiencyReport,
    deletion_flip_rate,
    sanity_check,
    sufficiency_keep_rate,
)
from neuro_co.xai.stability import (
    StabilityReport,
    load_explanation_traces,
    top_k_stability,
)

__version__ = "0.1.0"

__all__ = [
    "AttributionTrace",
    "ConceptBank",
    "ConceptFn",
    "DeletionReport",
    "InstanceRegistry",
    "SanityCheckReport",
    "StabilityReport",
    "SufficiencyReport",
    "__version__",
    "concept_registry",
    "contrastive_attribution",
    "deeplift_attribution",
    "deletion_flip_rate",
    "gradient_attribution",
    "infer_problem_name",
    "integrated_gradients",
    "load_explanation_traces",
    "register_concept_bank",
    "sanity_check",
    "sufficiency_keep_rate",
    "top_k_stability",
]
