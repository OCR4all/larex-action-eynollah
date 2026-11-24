# SPDX-License-Identifier: MIT
import math
import os
from pathlib import Path

CPU_QUOTA = Path('/sys/fs/cgroup/cpu.max')


def available_cpus(quota_file: Path = CPU_QUOTA) -> int:
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    try:
        quota, period = quota_file.read_text().split()
        cpus = min(cpus, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return max(1, cpus)
