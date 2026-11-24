# SPDX-License-Identifier: MIT
import pytest

from larex_action.parameters import InvalidParameters, LayoutOptions, parse_parameters


def test_defaults():
    assert parse_parameters({}) == LayoutOptions()


def test_layout_parameters():
    options = parse_parameters(
        {
            'fullLayout': False,
            'tables': True,
            'curvedLines': True,
            'readingOrder': 'model',
            'rightToLeft': True,
            'marginalia': 'left',
            'splitParagraphsAtMarginalia': True,
            'cropPage': False,
            'binarize': True,
            'regionSeparation': 0.3,
            'lineSeparation': 0.4,
            'polygonTolerance': 0,
        }
    )
    assert options == LayoutOptions(
        full_layout=False,
        tables=True,
        curved_lines=True,
        model_reading_order=True,
        right_to_left=True,
        marginalia='left',
        split_paragraphs_at_marginalia=True,
        crop_page=False,
        binarize=True,
        region_separation=0.3,
        line_separation=0.4,
        polygon_tolerance=0,
    )


def test_lines_mode_ignores_layout_parameters():
    options = parse_parameters({'mode': 'lines', 'tables': True, 'readingOrder': 'model', 'curvedLines': True})
    assert options.lines_only
    assert not (options.full_layout or options.tables or options.model_reading_order or options.curved_lines)


@pytest.mark.parametrize(
    ('parameters', 'expected'),
    [
        ({}, (0, 0)),
        ({'minColumns': 2}, (2, 6)),
        ({'maxColumns': 3}, (1, 3)),
        ({'minColumns': 2, 'maxColumns': 4}, (2, 4)),
    ],
)
def test_column_limits(parameters, expected):
    options = parse_parameters(parameters)
    assert (options.min_columns, options.max_columns) == expected


@pytest.mark.parametrize(
    'parameters',
    [
        {'minColumns': 4, 'maxColumns': 2},
        {'maxColumns': 7},
        {'mode': 'regions'},
        {'lineSeparation': 1.5},
        {'lineSeparation': float('nan')},
        {'marginalia': False},
    ],
)
def test_invalid_parameters(parameters):
    with pytest.raises(InvalidParameters, match='Invalid parameters'):
        parse_parameters(parameters)
