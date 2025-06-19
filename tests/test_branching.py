import math
import time

import matplotlib.pyplot as plt

from thesis import branching


def test_uniform():
    seq_len = 100
    len_distribution = [0] * (seq_len + 1)
    num_distribution = [0] * (seq_len + 1)

    num_trials = 1_000_000
    for _ in range(num_trials):
        num_branches = branching.uniform_y(seq_len=seq_len)
        branch_len = math.ceil(seq_len / num_branches)

        num_distribution[num_branches] += 1
        len_distribution[branch_len] += 1

    x = list(range(seq_len + 1))
    num_distribution = [val / num_trials for val in num_distribution]
    len_distribution = [val / num_trials for val in len_distribution]

    plt.figure(figsize=(10, 5))
    plt.bar(x, num_distribution, label="num_branches")
    plt.bar(x, len_distribution, label="branch_len")
    plt.xlabel("Key")
    plt.ylabel("Value")
    plt.title("Bar Plot of Dictionary Data")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_uniform()
