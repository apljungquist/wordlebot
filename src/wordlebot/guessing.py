import atexit
import collections
import dataclasses
import functools
import os

import more_itertools

FIRST_GUESS_ONLY = os.environ.get("FIRST_GUESS_ONLY") == "1"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _fmt_permitted(permitted):
    return "\n".join("".join(c if c in p else " " for c in ALPHABET) for p in permitted)


@dataclasses.dataclass(frozen=True)
class Constraint:
    permitted: tuple[tuple[str, ...], ...]
    lo: tuple[tuple[str, int], ...]
    hi: tuple[tuple[str, int], ...]

    @staticmethod
    def new_from_state(state):
        constraint = Constraint.new(ALPHABET)
        for guess, feedback in [step.split(":") for step in state.split(",")]:
            feedback = [int(f) for f in feedback]
            constraint = constraint.tightened(guess, feedback)
        return constraint

    @staticmethod
    def new(alphabet: str):
        return Constraint(
            permitted=tuple(tuple(alphabet) for _ in range(5)),
            lo=(),
            hi=(),
        )

    def tightened(self, guess, feedback):
        permitted = [set(p) for p in self.permitted]
        lo = collections.defaultdict(lambda: 0, self.lo)
        hi = collections.defaultdict(lambda: 5, self.hi)

        required = set()
        for i, (g, f) in enumerate(zip(guess, feedback)):
            match f:
                case 0:
                    assert g == "-"
                case 1:
                    permitted[i].discard(g)
                    # If a letter occurs multiple times in a guess but only once in the
                    # answer, only the first occurrence will be scored as a two.
                    if g not in required:
                        for p in permitted:
                            p.discard(g)
                case 2:
                    required.add(g)
                    permitted[i].discard(g)
                case 3:
                    required.add(g)
                    permitted[i] = {g}
                case _:
                    assert False

        positive = collections.Counter(
            g for g, f in zip(guess, feedback) if f in {2, 3}
        )
        negative = collections.Counter(g for g, f in zip(guess, feedback) if f in {1})
        for k, v in positive.items():
            lo[k] = max(lo[k], v)
            if k in negative:
                hi[k] = min(hi[k], v)

        return Constraint(
            permitted=tuple(tuple(p) for p in permitted),
            lo=tuple(lo.items()),
            hi=tuple(hi.items()),
        )

    def permits(self, word):
        for c, p in zip(word, self.permitted):
            if c not in p:
                return False

        counts = collections.Counter(word)
        for c, v in self.lo:
            if counts[c] < v:
                return False

        for c, v in self.hi:
            if v < counts[c]:
                return False

        return True


def _options(constraint, wordlist):
    """Return (superset of) possible answers"""
    # Superset because the information from the state may not be fully exploited
    return (word for word in wordlist if constraint.permits(word))


@functools.cache
def _choice(constraint, wordlist):
    """Return the word to try next

    Note that this need not be a possible answer.
    """
    return more_itertools.first(_options(constraint, wordlist))

atexit.register(lambda :print(_choice.__name__, _choice.cache_info()))

class Guesser:
    def __init__(self, wordlist: list[str]) -> None:
        self._wordlist = tuple(sorted(wordlist, key=lambda w: (-len(set(w)), w)))

    def __call__(self, state: str) -> str:
        if state == "-----:00000" and True:
            result = "tares"
        else:
            constraint = Constraint.new_from_state(state)
            result = _choice(constraint, self._wordlist)
        return result
