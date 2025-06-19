import math
import random
import sympy


def ceil_sqrt(seq_len: int) -> int:
    return math.ceil(math.sqrt(seq_len))


def floor_sqrt(seq_len: int) -> int:
    return max(0, math.floor(math.sqrt(seq_len)))


def uniform_y(seq_len: int) -> int:
    r = random.randint(1, seq_len)
    r = (r + (seq_len / r)) / 2
    return math.ceil(r)


def uniform_x(seq_len: int) -> int:
    return random.randint(1, seq_len)


def uniform_divisor(seq_len: int) -> int:
    return random.choice(sympy.divisors(seq_len))  # type: ignore


def branch_len_five(seq_len: int) -> int:
    return math.ceil(seq_len / 5)


def branch_len_four(seq_len: int) -> int:
    return math.ceil(seq_len / 4)


def branch_len_three(seq_len: int) -> int:
    return math.ceil(seq_len / 3)


def ceil_sqrt_rand_one(seq_len: int) -> int:
    return 1 if random.random() < 0.1 else ceil_sqrt(seq_len)
