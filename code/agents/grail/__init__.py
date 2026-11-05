"""GRAIL-style baseline for Tofu Kingdom.

This package ports the GRAIL architecture (Rahimirad et al., 2025,
"Bayesian Social Deduction with Graph-Informed Language Models",
https://github.com/shahabrrad/Bayesian-Avalon) to Tofu Kingdom:

- `pomegranate/`     : the belief-propagation engine vendored VERBATIM from the
                       official GRAIL repository (MIT licensed, see
                       pomegranate/LICENSE.bayesian-avalon). This is the exact
                       code GRAIL uses for max-product inference in Avalon.
- `tofu_factor_graph`: Tofu-Kingdom-specific factor graph construction
                       (the analogue of GRAIL's model_reduced_categories.py).

The GRAIL-style Prince agent itself (`GrailPrinceAgent`, the analogue of
GRAIL's agent_acl.py + policy_models/heuristic.py) lives in `prince_agent.py`;
its 'vibes' prompt is in `prompts.py` and the response schema in `schemas.py`
(both in this package).
"""
