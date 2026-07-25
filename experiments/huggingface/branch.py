import os
from collections.abc import Iterable
from typing import Any


def branch(values: Iterable, sequential: bool = False) -> Any:
    children = []

    for value in values:
        pid = os.fork()

        if pid == 0:
            return value

        if sequential:
            os.waitpid(pid, 0)
        else:
            children.append(pid)

    if not sequential:
        for pid in children:
            os.waitpid(pid, 0)

    os._exit(0)
