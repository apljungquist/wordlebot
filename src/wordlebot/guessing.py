import collections

import more_itertools

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _fmt_permitted(permitted):
    return "\n".join("".join(c if c in p else " " for c in ALPHABET) for p in permitted)


def _is_option(word, permitted, required):
    for c, p in zip(word, permitted):
        if c not in p:
            return False

    counts = collections.Counter(word)
    for c in required:
        if counts[c] < required[c]:
            return False

    return True


def _options(state, wordlist):
    """Return (superset of) possible answers"""
    # Superset because the information from the state may not be fully exploited
    permitted = [set(ALPHABET) for _ in range(5)]
    required = collections.defaultdict(int)
    for guess, feedback in [step.split(":") for step in state.split(",")]:
        marginal_required = collections.defaultdict(int)
        for i, (g, f) in enumerate(zip(guess, feedback)):
            f = int(f)
            match f:
                case 0:
                    assert g == "-"
                case 1:
                    permitted[i].discard(g)
                    # If a letter occurs multiple times in a guess but only once in the
                    # answer, only the first occurrence will be scored as a two.
                    if g not in marginal_required:
                        for p in permitted:
                            p.discard(g)
                case 2:
                    marginal_required[g] += 1
                    permitted[i].discard(g)
                case 3:
                    marginal_required[g] += 1
                    permitted[i] = {g}
                case _:
                    assert False

        for k, v in marginal_required.items():
            required[k] = max(required[k], v)

    for word in wordlist:
        if _is_option(word, permitted, marginal_required):
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
