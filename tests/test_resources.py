# SPDX-License-Identifier: MIT
import os

import pytest

from larex_action.resources import available_cpus


@pytest.mark.parametrize(('quota', 'expected'), [('100000 100000', 1), ('max 100000', None), ('invalid', None)])
def test_cpu_quota(tmp_path, quota, expected):
    file = tmp_path / 'cpu.max'
    file.write_text(quota)
    assert available_cpus(file) == (expected or available_cpus(tmp_path / 'missing'))


def test_without_quota(tmp_path):
    assert 1 <= available_cpus(tmp_path / 'missing') <= os.cpu_count()
