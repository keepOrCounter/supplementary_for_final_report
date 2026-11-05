from __future__ import annotations

import random

import numpy as np
from numpy.random import SeedSequence


class SeedSource:
    """A node in a deterministic seed tree (see module docstring).
    Hierarchical, collision-free seeding for reproducible + comparable experiments.

        - Reproducibility: the same root seed always produces the same streams.
        - Isolation: one component drawing more / fewer seeds (e.g. an LLM agent
        retrying a variable number of times) never shifts another component's
        stream.
        - Nesting: ``spawn`` children for Monte-Carlo rollouts or nested LLM calls,
        each an independent stream, without ever computing a "spacing" offset.

    Usage contract
    --------------
    A node hands out randomness through two channels that share one internal
    counter, so they can be freely interleaved on the same node without overlap:

        - ``next_seed()``   -> a fresh 32-bit int seed (for an LLM API ``seed=`` arg)
        - ``to_random()``  -> a seeded ``random.Random`` (for random choices)

    For genuinely independent concerns within one component (e.g. an agent that
    both makes random guesses AND seeds LLM calls), ``spawn(key)`` a separate
    child per concern. Then changing one concern's draw count cannot disturb the
    other -- exactly the isolation we want when toggling ablation strategies.

    ``SeedSource(None)`` is a "null" source: ``next_seed()`` returns ``None`` (so the
    LLM call stays seedless / nondeterministic) and ``to_random()`` returns an
    unseeded ``random.Random``. This preserves the original ``seed=None`` semantics.
    """

    __slots__ = ("__ssequence", "__used_seeds")

    def __init__(self, seed: int | SeedSequence | None):
        """Create a seed source.

        Args:
            seed: an int root seed, an existing ``SeedSequence`` (used internally
                by ``spawn`` to wrap a derived child), or ``None`` for a null
                (nondeterministic) source.
        """
        if seed is None:
            # Null mode: no underlying SeedSequence at all.
            self.__ssequence: SeedSequence | None = None
        elif isinstance(seed, SeedSequence):
            # Already a SeedSequence (a child handed back by spawn) -> wrap as-is.
            self.__ssequence = seed
        else:
            # An int root seed becomes the SeedSequence's `entropy`. From here on,
            # this object's entire stream is determined by that one number.
            self.__ssequence = SeedSequence(int(seed))
        self.__used_seeds = 0 # How many 32-bit words next_seed/to_random have consumed from THIS node's
        # own stream.

    @property
    def is_null(self) -> bool:
        """True if this source produces no seed (nondeterministic mode)."""
        return self.__ssequence is None

    def spawn(self, key: int) -> SeedSource:
        """Return an independent child stream identified by ``key``.

        The child depends only on (root entropy, key path). So ``spawn(2)`` always yields
        the same stream regardless of surrounding calls.
        """
        if self.__ssequence is None:
            # A null source spawns null children (None propagates down the tree).
            return SeedSource(None)
        # Build the child by hand instead of using the official `.spawn()`.
        # Official spawn assigns keys from a counter (n_children_spawned), making
        # it order-dependent; we instead APPEND our chosen `key` to this node's
        # spawn_key path. Same root `entropy`, deeper `spawn_key`:
        #   root ()  --spawn(2)-->  (2,)  --spawn(5)-->  (2, 5)
        # numpy then hashes (entropy, spawn_key) into the child's pool, giving a
        # fresh, independent, fully reproducible stream keyed by the path alone.
        child = SeedSequence(
            entropy=self.__ssequence.entropy, # same root number
            spawn_key=tuple(self.__ssequence.spawn_key) + (int(key),), # this node's path + key
        )
        return SeedSource(child)

    def __draw_word(self) -> int:
        """Pull the next 32-bit word from this node's own stream.
        """
        word = int(self.__ssequence.generate_state(self.__used_seeds + 1)[self.__used_seeds])
        self.__used_seeds += 1
        return word

    def next_seed(self) -> "int | None":
        """Next 32-bit int seed for an LLM call, or ``None`` if null.

        ``int(...)`` in __draw_word ensures a builtin ``int`` (not ``np.uint32``),
        which is what the OpenAI/Ollama ``seed=`` arg and JSON logging expect.
        """
        if self.__ssequence is None:
            return None
        return self.__draw_word()

    def to_random(self) -> random.Random:
        """A ``random.Random`` seeded from this node's stream.

        Consumes 4 words (128 bits) via the same counter as ``next_seed`` so the
        two channels stay in lock-step on one node (no overlap); an unseeded
        ``Random`` in null mode.
        """
        if self.__ssequence is None:
            return random.Random()
        # Draw four 32-bit words and pack them into one 128-bit integer seed,
        # so random.Random is seeded with the full entropy of this stream.
        words = [self.__draw_word() for _ in range(4)]
        seed_int = int.from_bytes(np.array(words, dtype=np.uint32).tobytes(), "little")
        return random.Random(seed_int)


def player_id_from_answer(answer: str, center_id: int = -1) -> int:
    if answer == "center":
        return center_id
    return int(answer.split()[1])  # answers like "Player 3"


def action_description(action: dict) -> str:
    """more human friendly action description"""
    action_type = action["type"]
    
    if action_type == "question":
        question = action["question_type"]
        target = action["target"]
        param = action.get("question_param")
        description = f"Ask Player {target}: '{question}'"
        
        if param is not None and param != -1:
            description += f" (about Player {param})"
        elif param == -1:
            description += " (about center card)"
        return description
    
    elif action_type == "answer":
        return f"Answer: \"{action['answer']}\""
    
    elif action_type == "guess":
        target = action["target"]
        if target == -1:
            return "Guess: center card"
        return f"Guess: Player {target}"
    
    else:
        return str(action)