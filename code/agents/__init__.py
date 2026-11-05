"""Agent implementations for Tofu Kingdom, one sub-package per method:

- `base`        : BaseAgent interface, RandomAgent, HumanAgent, shared agent exceptions.
- `llm_agent`   : plain LLM baseline agent (no explicit belief state).
- `our_agents/` : our method -- exact Kripke belief tracking + information-gain-driven
                  active question selection, its soft-likelihood variant, and the
                  ablation Prince.
- `grail/`      : GRAIL baseline (Rahimirad et al., 2025), factor-graph BP + passive policy.
- `emo/`        : EMO baseline (Yu, Zhang & Lu, NAACL 2025), explicit LLM opponent models.
- `prompts`     : prompt templates shared by every LLM-driven agent (game rules,
                  generic action prompts); method-specific prompts live in each
                  method's own `prompts.py`.

Every public agent class is re-exported here, so callers can simply
`from agents import <AgentClass>`.
"""
from agents.base import BaseAgent, RandomAgent, HumanAgent, LLMParseError, SymbolicGuessError, \
    BeliefUpdateError
from agents.llm_agent import LLMAgent
from agents.our_agents.symbolic_agents import SymbolicPrinceAgent, SoftSymbolicPrinceAgent, \
    SymbolicPlayerAgent, FirstOrderSymbolicPlayerAgent
from agents.our_agents.ablation_agent import AblationPrinceAgent
from agents.emo.prince_agent import EMOPrinceAgent

__all__ = [
    "BaseAgent", "RandomAgent", "HumanAgent", "LLMParseError", "SymbolicGuessError",
    "BeliefUpdateError",
    "LLMAgent",
    "SymbolicPrinceAgent", "SoftSymbolicPrinceAgent", "SymbolicPlayerAgent",
    "FirstOrderSymbolicPlayerAgent",
    "AblationPrinceAgent",
    "GrailPrinceAgent",
    "EMOPrinceAgent",
]


def __getattr__(name: str):
    """import GrailPrinceAgent on the first read of it

    It is the one re-export not imported above. Importing it loads torch and a
    belief propagation engine, which costs seconds and a few hundred megabytes,
    and every other import of this package would otherwise pay for it. The
    Resistance runner is the reason it matters: reaching agents.resistance runs
    this file, so a random-only Resistance run was loading torch.

    It still reads as an ordinary attribute: `from agents import GrailPrinceAgent`
    and `agents.GrailPrinceAgent` both import it on the spot.
    """
    if name == "GrailPrinceAgent":
        from agents.grail.prince_agent import GrailPrinceAgent
        return GrailPrinceAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
