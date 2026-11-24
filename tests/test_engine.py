# SPDX-License-Identifier: MIT
import os
from io import BytesIO
from pathlib import Path

import ocrd_validators
import pytest
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

from larex_action.parameters import parse_parameters
from larex_action.resources import available_cpus

pytestmark = pytest.mark.models

SCHEMA = etree.XMLSchema(etree.parse(Path(ocrd_validators.__file__).parent / 'page.xsd'))
PAGE = '{http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15}'
LINE = 'The quick brown fox jumps over the lazy dog and runs away'


@pytest.fixture(scope='module')
def engine():
    from larex_action.engine import DeviceSettings, LayoutEngine

    return LayoutEngine(Path(os.environ['EYNOLLAH_MODEL_DIR']), DeviceSettings('cpu', available_cpus()))


@pytest.fixture(scope='module')
def scan():
    image = Image.new('RGB', (1240, 1754), (238, 232, 220))
    draw = ImageDraw.Draw(image)
    draw.text((430, 130), 'Chapter One', fill=(30, 25, 20), font=ImageFont.load_default(64))
    for paragraph in range(3):
        for line in range(8):
            position = (140, 300 + paragraph * 430 + line * 44)
            draw.text(position, LINE, fill=(30, 25, 20), font=ImageFont.load_default(34))
    return image


def analyze(engine, scan, parameters):
    result = engine.process(scan, 'scan.png', parse_parameters(parameters))
    tree = etree.parse(BytesIO(result))
    SCHEMA.assertValid(tree)
    return tree


@pytest.mark.parametrize(
    'parameters',
    [
        {},
        {'fullLayout': False, 'marginalia': 'off', 'cropPage': False},
        {'tables': True, 'readingOrder': 'model'},
        {'binarize': True, 'rightToLeft': True, 'polygonTolerance': 0},
        {'curvedLines': True, 'maxColumns': 1},
    ],
    ids=['default', 'simple', 'tables-model-order', 'binarize', 'curved'],
)
def test_layout(engine, scan, parameters):
    tree = analyze(engine, scan, parameters)
    regions = tree.findall(f'.//{PAGE}TextRegion')
    assert regions
    assert 15 <= len(tree.findall(f'.//{PAGE}TextLine')) <= 40
    references = [ref.get('regionRef') for ref in tree.iterfind(f'.//{PAGE}RegionRefIndexed')]
    assert references
    assert len(set(references)) == len(references)


def test_lines(engine, scan):
    tree = analyze(engine, scan, {'mode': 'lines'})
    assert len(tree.findall(f'.//{PAGE}TextRegion')) == 1
    assert 15 <= len(tree.findall(f'.//{PAGE}TextLine')) <= 40


def test_blank_page(engine):
    blank = Image.new('RGB', (800, 1100), (240, 240, 240))
    assert engine.process(blank, 'blank.png', parse_parameters({})) is None


def test_tiny_image(engine):
    assert engine.process(Image.new('RGB', (40, 30), (240, 240, 240)), 'page.png', parse_parameters({})) is None
