"""Our method for Tofu Kingdom: exact Bayesian/Kripke belief tracking +
information-gain-driven *active* question selection (+ LLM-modelled opponents).

- `Kripke`          : weighted Kripke structure (possible worlds + Bayesian update +
                      information gain), and its neuro-symbolic variant with the
                      LLM-judged soft answer likelihood.
- `symbolic_agents` : SymbolicPrinceAgent (IG question selection + MAP guess),
                      SoftSymbolicPrinceAgent (Top-k soft IG), SymbolicPlayerAgent
                      (strategic non-prince opponent).
- `ablation_agent`  : AblationPrinceAgent (question / guess / likelihood all configurable).
- `prompts`         : soft-likelihood judge prompt + ablation Prince prompt.
- `schemas`         : structured LLM outputs for the soft-likelihood judge.
"""
