import math

from thesis import branching


def compute_average_branch_len(N: int = 40):
    total_branch_len = 0
    for seq_len in range(1, N + 1):
        num_branches = branching.ceil_sqrt(seq_len)
        branch_len = math.ceil(seq_len / num_branches)
        total_branch_len += branch_len

    return total_branch_len / N


if __name__ == "__main__":
    print(compute_average_branch_len())
