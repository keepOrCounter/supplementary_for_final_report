"""EMO baseline for Tofu Kingdom.

A faithful port of EMO (Yu, Zhang & Lu, "LLM-Based Explicit Models of Opponents
for Multi-Agent Games", NAACL 2025) as a passive-opponent-modelling Prince
baseline. The paper ships no code, so this is a from-the-paper reimplementation:

- `schemas` : structured LLM outputs (the Tofu analogue of the paper's Appendix-D list format).
- `prompts` : the EXPLICIT-MODEL-OF-OPPONENTS and AGENT prompt groups of
    Appendix D, adapted to Tofu Kingdom (ENVIRONMENT is reused
    verbatim from the shared `agents/prompts.py`).
- `opponent_model` : `AtomicOpponentModel` (Eqs. 1-3, 5, memory-enhanced self-refine) and `CentralValidator` (Eqs. 4, 6).

The EMO Prince agent itself (`EMOPrinceAgent`) lives in `prince_agent.py`
alongside these components, and drives them through
Algorithm 1 (per-opponent update) and Algorithm 2 (CoT + EMO). The mapping from
the Prince to the paper's agent i, and from each player to an opponent j, is
documented on that class and in the README.
"""
