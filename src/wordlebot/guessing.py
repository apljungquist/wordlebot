import collections

import more_itertools

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _fmt_permitted(permitted):
    return "\n".join("".join(c if c in p else " " for c in ALPHABET) for p in permitted)


def _is_option(word, permitted, lo, hi):
    for c, p in zip(word, permitted):
        if c not in p:
            return False

    counts = collections.Counter(word)
    for c, v in lo.items():
        if counts[c] < v:
            return False

    for c, v in hi.items():
        if v < counts[c]:
            return False

    return True


def _options(state, wordlist):
    """Return (superset of) possible answers"""
    # Superset because the information from the state may not be fully exploited
    permitted = [set(ALPHABET) for _ in range(5)]
    lo = collections.defaultdict(lambda: 5)
    hi = collections.defaultdict(lambda: 0)
    for guess, feedback in [step.split(":") for step in state.split(",")]:
        feedback = [int(f) for f in feedback]
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
            hi[k] = max(hi[k], v)
            if k in negative:
                lo[k] = min(lo[k], v)

    for word in wordlist:
        if _is_option(word, permitted, hi, lo):
            yield word


def _choice(state, wordlist):
    """Return the word to try next

    Note that this need not be a possible answer.
    """
    return more_itertools.first(_options(state, wordlist))


class Guesser:
    def __init__(self, wordlist: list[str]) -> None:
        self._wordlist = sorted(wordlist, key=lambda w: (-len(set(w)), w))

    def __call__(self, state: str) -> str:
        return _choice(state, self._wordlist)
