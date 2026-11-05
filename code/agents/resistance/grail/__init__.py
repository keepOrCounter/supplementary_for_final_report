"""GRAIL's Graph Only ablation for The Resistance.

Ported from the official repository (Rahimirad et al., 2025, "Bayesian Social
Deduction with Graph-Informed Language Models",
https://github.com/shahabrrad/Bayesian-Avalon), MIT licensed, see
LICENSE.bayesian-avalon.

Vendored from upstream, every edit marked with `[Resistance port]`:
- `model_reduced_categories`: the factor graph over six roles and five missions
- `base_model` : the graph's base class
- `heuristic` : the belief greedy policy
- `models/v2/` : the trained factor weights

Written here:
- `ateam` : GRAIL's team enum, lifted out of their agent_base.py
- `state_vector`: our mission records to their 21 entry state vector
- `agent` : the agent itself, replacing their agent_acl.py

The belief propagation engine is not vendored twice; `agents/grail/pomegranate`
is the same upstream code, already vendored for the Tofu Kingdom port.
"""
