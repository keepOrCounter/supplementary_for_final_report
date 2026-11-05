# [Resistance] our method: what one mission is worth, read off the score alone.
#
# New, with no Tofu Kingdom counterpart, because the two games value an action
# differently. A Tofu question costs nothing: only the final guess has a payoff,
# so "ask whatever is most informative, then guess" is close to optimal. Here
# every action is a payoff action and an information action at once, and a
# resistance player that only chases information loses the game while learning.
#
# The state space is small enough to solve exactly. Only the score matters, and
# there are nine reachable scores, so the value of every one of them is a table
# computed once by backward induction and then read.
from __future__ import annotations

from game.resistance_data_classes import MISSIONS_TO_WIN


# The chance a mission succeeds when nothing is known about the team. It is the
# one input the table needs, and the ordering it produces is not sharply
# sensitive to it: 2 successes against 2 failures is the most valuable mission at
# every value, and 2 against 0 the least. The middle of the table does move, so
# it is worth sweeping.
MISSION_SUCCESS = 0.6

_TABLES: dict[float, dict[tuple[int, int], float]] = {}


def win_values(mission_success = None) -> dict[tuple[int, int], float]:
    """P(the resistance eventually wins) for every score, by backward induction

    The value function of a Markov reward process. No action, so its Bellman
    equation is an expectation and not a max:

        V(s) = p V(one more success) + (1 - p) V(one more failure)
            p   mission_success at this score, the transition probability

    Solved in one backward sweep and not iterated, because the score graph is
    acyclic. See docs/standard_names.md.

    Args:
        mission_success: P(a mission succeeds), either one number used for every
            mission or one number per mission, indexed by missions played. Per
            mission is worth having: a six player game sends teams of 2, 3, 4, 3
            and 4, and a bigger team is far likelier to carry a spy, so no single
            number describes all five. The callers read it off the belief, which
            is what keeps it from being a hyperparameter.

    Keyed by (missions succeeded, missions failed). A score with MISSIONS_TO_WIN
    successes is a win and one with that many failures is a loss, and every other
    score is worth what the next mission leads to.

    Built once per value of mission_success and then cached, because it is nine
    numbers that never change during a run.
    """
    # Resolved here and not in the signature, because a default argument is bound
    # once when the function is defined: with MISSION_SUCCESS written there, a
    # sweep or a test that reassigns the module constant would be ignored, and
    # silently so.
    mission_success = MISSION_SUCCESS if mission_success is None else mission_success
    if isinstance(mission_success, (int, float)):
        mission_success = (mission_success,) * (2 * MISSIONS_TO_WIN - 1)
    mission_success = tuple(mission_success)
    for chance in mission_success:
        if not 0.0 <= chance <= 1.0:
            raise ValueError(f"mission_success must be probabilities, got {mission_success}")
    if len(mission_success) < 2 * MISSIONS_TO_WIN - 1:
        raise ValueError(f"mission_success needs one entry per mission that can be played, "
            f"{2 * MISSIONS_TO_WIN - 1} of them, got {len(mission_success)}")

    cached = _TABLES.get(mission_success)
    if cached is not None:
        return cached

    values = {}
    # Backwards: a score is only worth what the scores after it are worth, and
    # those all have one more mission played.
    # From the last score any game can still be read out of, which is one mission
    # short of both sides having won: nothing ever asks what MISSIONS_TO_WIN
    # successes against MISSIONS_TO_WIN failures is worth, because that score
    # cannot happen.
    for played in range(2 * MISSIONS_TO_WIN - 1, -1, -1): # mission index
        for successes in range(MISSIONS_TO_WIN + 1):
            failures = played - successes
            if not 0 <= failures <= MISSIONS_TO_WIN:
                continue
            if successes == MISSIONS_TO_WIN:
                values[(successes, failures)] = 1.0
            elif failures == MISSIONS_TO_WIN:
                values[(successes, failures)] = 0.0
            else:
                # The chance of the mission this score is about to play, which
                # is the one at this many missions played.
                chance = mission_success[played]
                values[(successes, failures)] = chance * values[(successes + 1, failures)] \
                    + (1.0 - chance) * values[(successes, failures + 1)]

    _TABLES[mission_success] = values
    return values


def mission_value(successes: int, failures: int, succeeds: bool, mission_success: float = None) -> float:
    """P(eventually win rate | one more success) or P(eventually win rate | one more failure) from this score, from 0 to 1

    What the game is worth to the resistance after this mission goes that way.
    Raises on a score that has already won or lost, because there is no next
    mission to be worth anything.
    """
    values = win_values(mission_success)
    return values[(successes + 1, failures)] if succeeds else values[(successes, failures + 1)]


def team_value(successes: int, failures: int, succeeds: float, mission_success = None) -> float:
    """what running a team with this chance of succeeding is worth, in win probability
        It is the Q function of the team choice, and the one thing every player in this
        
        succeeds V(one more success) + (1 - succeeds) V(one more failure)
            succeeds  P(this team's mission succeeds), whoever is asking
            V(x)      P(the resistance eventually wins) at score x

    The one thing every player in this game works out about a team, and the
    reason it lives here rather than in either caller: a level k opponent model
    is a level k-1 player's decision, so the belief that models a player and the
    agent that is one score a team with the same line. They differ in what they
    put in succeeds, which is the whole of what knowing your own card buys.

    The Q function of the team choice, and fed a belief-averaged succeeds it is
    QMDP: affine in succeeds, so averaging Q over the worlds and taking Q at the
    averaged chance agree to 1.1e-16. The continuation table is shared rather
    than one per world, which is certainty equivalence on top, worth 6e-4.

    QMDP never pays to learn, which is what info_weight is for. See
    docs/standard_names.md.

    Written for the resistance; a spy wants one minus it.
    """
    values = win_values(mission_success)
    return succeeds * values[(successes + 1, failures)] \
        + (1.0 - succeeds) * values[(successes, failures + 1)]


def team_gain(successes: int, failures: int, succeeds: float, mission_success = None) -> float:
    """the same, less what an average team is worth, in missions
    
    It is the advantage function of the team choice, and the one thing every player
    A(s, a) = Q(s, a) - V(s)

        gain = (team_value(...) - V(the score now)) / swing(the score now)
            = succeeds - p
            p   P(an average team succeeds) at this mission, from mission_success

    The two lines are equal because V(the score now) is by construction
    p V(one more success) + (1 - p) V(one more failure), so swing cancels.

    Zero means an average team, positive means better than the value table
    assumed. Read this where a number has to be dimensionless, which is where a
    softmax sharpness has to mean the same thing at every score; read team_value
    where win probability is already the unit.

    The advantage function A(s, a) = Q(s, a) - V(s), over swing. swing is
    dQ/d(succeeds), so the quotient is the advantage in units of mission success
    probability, which is the same unit at every score.
    """
    values = win_values(mission_success)
    above = team_value(successes, failures, succeeds, mission_success) - values[(successes, failures)]
    return above / max(swing(successes, failures, mission_success), 1e-9)


def swing(successes: int, failures: int, mission_success: float = None) -> float:
    """P(eventually win rate | one more success) - P(eventually win rate | one more failure), from 0 to 1

    Means how much win rate decreases if let mission failed rather than succeed, smaller means
    the mission is less important to win, worth for gathering information

    A difference between two probabilities and not a probability itself, though
    it lands in 0 to 1 either way: winning a mission never leaves the resistance
    worse off, so the difference is never negative, and both terms are at most 1.
    That is what lets an information weight be scaled by 1 - swing.

    The number a one step value cannot see. The same mission is worth 6.25 times
    as much at 2 successes against 2 failures as it is at 2 against 0, and a
    policy that weighs the two the same is blind to the score.

    Also what makes an information weight adaptive: information is cheap when the
    swing is small, because there is room to lose the mission and learn from it,
    and there is no room at all when the swing is 1.

    It is dQ/d(succeeds): team_value is affine in that argument with this slope.
    """
    return mission_value(successes, failures, True, mission_success) \
        - mission_value(successes, failures, False, mission_success)
