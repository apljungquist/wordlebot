import atexit
import collections
import dataclasses
import functools
import operator
import time
from math import log

import more_itertools
from botfights import load_wordlist

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


def _min_surprise(options, guess):
    """Return entropy of the score"""
    counter = collections.Counter(_quick_score(secret, guess) for secret in options)
    numerator = max(counter.values())
    denominator = sum(counter.values())
    return log(denominator / numerator)


@functools.cache
def _options(constraint, wordlist):
    """Return (superset of) possible answers"""
    # Superset because the information from the state may not be fully exploited
    return [word for word in wordlist if constraint.permits(word)]


atexit.register(lambda: print(_options.__name__, _options.cache_info()))


@functools.cache
def _choice(constraint, allowed_guesses, allowed_answers, adversarial):
    """Return the word to try next

    Note that this need not be a possible answer.
    """
    plausible_answers = _options(constraint, allowed_answers)
    # If there are only three options left and we guess at random then we expect to use
    # two more guesses. If we first guess a word that is impossible then we will need
    # at least two guesses. As such, switching to choosing only from plausible answers
    # will not hurt.
    if len(plausible_answers) <= 3:
        plausible_guesses = plausible_answers
    else:
        plausible_guesses = allowed_guesses

    if adversarial:
        rating = _min_surprise
    else:
        rating = _entropy

    ratings = {guess: rating(plausible_answers, guess) for guess in plausible_guesses}

    # Ordered collection before this point for reproducibility
    plausible_answers = set(plausible_answers)
    return max(ratings, key=lambda k: (ratings[k], k in plausible_answers))


atexit.register(lambda: print(_choice.__name__, _choice.cache_info()))


class SimpleGuesser:
    def __init__(self, wordlist: dict[str, bool]) -> None:
        self._guesses = tuple(sorted(wordlist))
        self._answers = tuple(sorted(k for k, v in wordlist.items() if v))

    def __call__(self, state: str) -> str:
        constraint = Constraint.new_from_state(state)
        return more_itertools.first_true(self._answers, pred=constraint.permits)


class MaxEntropyGuesser(SimpleGuesser):
    def __call__(self, state: str) -> str:
        constraint = Constraint.new_from_state(state)
        result = _choice(constraint, self._guesses, self._answers, False)
        return result


class MaximinSurpriseGuesser(SimpleGuesser):
    def __call__(self, state: str) -> str:
        constraint = Constraint.new_from_state(state)
        return _choice(constraint, self._guesses, self._answers, True)


class CheapHeuristicGuesser(SimpleGuesser):
    # cheap here means it can be precomputed
    def __init__(self, wordlist: dict[str, bool]) -> None:
        super().__init__(wordlist)
        self._answers = sorted(self._answers, key=lambda g: len(set(g)), reverse=True)


def _play(bot, answer):
    state = "-----:00000"
    guess = None
    i = 10
    while guess != answer:
        guess = bot(state)
        state += f",{guess}:{''.join(map(str, _quick_score(answer, guess)))}"
        i -= 1
        if not i:
            raise Exception
    return state.split(",", maxsplit=1)[1]


def _histogram(bot, answers):
    return dict(
        sorted(
            collections.Counter(
                _play(bot, answer).count(",") + 1 for answer in answers
            ).items(),
            key=operator.itemgetter(1),
        )
    )


def args_cache(func):
    func.cache = {}
    func.cache_hits = 0

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if args in func.cache:
            func.cache_hits += 1
            return func.cache[args]
        result = func(*args, **kwargs)
        func.cache[args] = result
        return result

    atexit.register(
        lambda: print(
            func.__qualname__, "Hits:", func.cache_hits, "Misses:", len(func.cache)
        )
    )
    return wrapped


def cache(func):
    result = functools.cache(func)
    atexit.register(lambda: print(func.__qualname__, result.cache_info()))
    return result


class NoSolutionError(Exception):
    pass


class StrategicGuesser:
    def __init__(self, wordlist: dict[str, bool], beam_width, max_depth) -> None:
        self.guesses = tuple(sorted(wordlist))
        self.answers = tuple(sorted(k for k, v in wordlist.items() if v))
        self.beam_width = beam_width
        self.max_depth = max_depth

    @args_cache
    def _guess(self, clues, *, plausible_answers):
        used = {clue[0] for clue in clues}
        beam = sorted(
            [guess for guess in self.guesses if guess not in used],
            key=lambda guess: (
                _entropy(plausible_answers, guess),
                guess in plausible_answers,
            ),
            reverse=True,
        )[: self.beam_width]

        max_cost = self.max_depth * len(self.answers)
        min_cost = len(clues) + 1

        best_cost = max_cost
        best_guess = None
        for guess in beam:
            try:
                cost = self._cost(clues, guess, old_plausible_answers=plausible_answers)
            except NoSolutionError:
                continue

            # Cannot do any better so why try.
            # I think this happens iff the guess is correct.
            # 25x speedup.
            if cost == min_cost:
                return cost, guess

            if cost < best_cost:
                best_cost = cost
                best_guess = guess

        if best_guess is None:
            raise NoSolutionError

        return best_cost, best_guess

    def _cost(self, clues, guess, *, old_plausible_answers):
        depth = len(clues) + 1
        if self.max_depth < depth:
            raise NoSolutionError

        scores = more_itertools.map_reduce(
            old_plausible_answers,
            keyfunc=lambda answer: _quick_score(answer, guess),
        )
        costs = []
        for i, (score, new_plausible_answers) in enumerate(scores.items()):
            if score == (3,) * 5:
                cost = depth
            else:
                cost, _ = self._guess(
                    clues | {(guess, score)}, plausible_answers=new_plausible_answers
                )
            costs.append(cost)

        return sum(costs)

    @cache
    def __call__(self, state):
        constraint = Constraint.new_from_state(state)
        _, guess = self._guess(
            frozenset(
                tuple(part.split(":")) for part in state.split(",") if part[0] != "-"
            ),
            plausible_answers=[
                answer for answer in self.answers if constraint.permits(answer)
            ],
        )
        return guess


def main():
    words = load_wordlist("bot", 0.5 ** 8)
    bots = [
        MaxEntropyGuesser(words),
        StrategicGuesser(words, 5, 7),
    ]
    for bot in bots:
        t = time.perf_counter()
        result = _histogram(bot, [k for k, v in words.items() if v])
        d = time.perf_counter() - t
        print(sum(k * v for k, v in result.items()), dict(sorted(result.items())), d)


if __name__ == "__main__":
    main()
