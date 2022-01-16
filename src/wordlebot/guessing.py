import atexit
import collections
import dataclasses
import functools
import operator
import os
from math import log

# Using a precomputed first guess is relatively safe and provides a massive speedup.
# Set this flag to redo that computation and exit.
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


def _quick_score(secret, guess):
    result = [None] * 5
    remaining = list(secret)

    for i, (s, g) in enumerate(zip(secret, guess)):
        if s == g:
            result[i] = 3
            remaining[i] = None

    for i, g in enumerate(guess):
        if result[i]:
            continue

        if g in remaining:
            result[i] = 2
            remaining[remaining.index(g)] = None
        else:
            result[i] = 1

    return tuple(result)


def _entropy(options, guess):
    """Return entropy of the score"""
    counter = collections.Counter(_quick_score(secret, guess) for secret in options)
    denominator = sum(counter.values())
    return -sum(
        numerator / denominator * log(numerator / denominator)
        for numerator in counter.values()
        if numerator
    )


@functools.cache
def _options(constraint, wordlist):
    """Return (superset of) possible answers"""
    # Superset because the information from the state may not be fully exploited
    return [word for word in wordlist if constraint.permits(word)]


atexit.register(lambda: print(_options.__name__, _options.cache_info()))


@functools.cache
def _choice(constraint, words):
    """Return the word to try next

    Note that this need not be a possible answer.
    """
    options = _options(constraint, words)
    # If there are only three options left and we guess at random then we expect to use
    # two more guesses. If we first guess a word that is impossible then we will need
    # at least two guesses. As such, switching to choosing only from possible words
    # will not hurt and may help.
    if len(options) <= 3:
        guesses = options
    else:
        guesses = words

    entropies = {guess: _entropy(options, guess) for guess in guesses}

    if FIRST_GUESS_ONLY:
        print(max(entropies.items(), key=operator.itemgetter(1)))
        exit()

    return max(entropies, key=entropies.__getitem__)


atexit.register(lambda: print(_choice.__name__, _choice.cache_info()))


class Guesser:
    def __init__(self, wordlist: list[str]) -> None:
        self._wordlist = frozenset(wordlist)

    def __call__(self, state: str) -> str:
        if state == "-----:00000" and not FIRST_GUESS_ONLY:
            result = "tares"
        else:
            constraint = Constraint.new_from_state(state)
            result = _choice(constraint, self._wordlist)
        return result
