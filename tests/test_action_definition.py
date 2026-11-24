# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from larex_action.parameters import RunParameters

ACTION = yaml.safe_load((Path(__file__).parents[1] / 'action.yml').read_text())
PARAMETERS = ACTION['parameters']


def test_parameters_match_the_implementation():
    fields = RunParameters.model_fields
    assert set(PARAMETERS) == {field.alias for field in fields.values()}
    for field in fields.values():
        assert PARAMETERS[field.alias]['default'] == field.default


@pytest.mark.parametrize('name', [name for name, value in PARAMETERS.items() if 'allowedValues' in value])
def test_allowed_values_are_accepted(name):
    for choice in PARAMETERS[name]['allowedValues']['values']:
        assert isinstance(choice['value'], str)
        RunParameters.model_validate({name: choice['value']})


@pytest.mark.parametrize('name', [name for name, value in PARAMETERS.items() if 'max' in value])
def test_numeric_ranges_match(name):
    RunParameters.model_validate({name: PARAMETERS[name]['min']})
    RunParameters.model_validate({name: PARAMETERS[name]['max']})
    with pytest.raises(ValidationError):
        RunParameters.model_validate({name: PARAMETERS[name]['min'] - 1})
    with pytest.raises(ValidationError):
        RunParameters.model_validate({name: PARAMETERS[name]['max'] + 1})
