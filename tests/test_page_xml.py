# SPDX-License-Identifier: MIT
import math
from io import BytesIO
from pathlib import Path

import ocrd_validators
import pytest
from eynollah.utils.xml import create_page_xml, xml_reading_order
from lxml import etree
from ocrd_models.ocrd_page import (
    AlternativeImageType,
    BorderType,
    CoordsType,
    ImageRegionType,
    TextLineType,
    TextRegionType,
)
from shapely import Polygon

from larex_action.page_xml import CREATOR, PageMetadata, clean_polygon, finalize
from larex_action.parameters import LayoutOptions

NAMESPACE = {'pc': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}
SCHEMA = etree.XMLSchema(etree.parse(Path(ocrd_validators.__file__).parent / 'page.xsd'))
SQUARE = '10,10 90,10 90,90 10,90'


def document(*regions, border='0,0 100,0 100,100 0,100'):
    result = create_page_xml('internal.png', 100, 100)
    page = result.get_Page()
    if border:
        page.set_Border(BorderType(Coords=CoordsType(points=border)))
    for region in regions:
        getattr(page, f'add_{type(region).__name__.removesuffix("Type")}')(region)
    return result


def text_region(identifier='region_0001', points=SQUARE, lines=(SQUARE,), conf=0.9):
    region = TextRegionType(id=identifier, type_='paragraph', Coords=CoordsType(points=points, conf=conf))
    for index, line in enumerate(lines):
        region.add_TextLine(TextLineType(id=f'{identifier}_line_{index}', Coords=CoordsType(points=line)))
    return region


def build(source, options=None, file_name='scan 1.png'):
    xml = finalize(source, PageMetadata(file_name, 100, 100), options or LayoutOptions())
    if xml is None:
        return None
    tree = etree.parse(BytesIO(xml))
    SCHEMA.assertValid(tree)
    return tree


def find(tree, path):
    return tree.xpath(path, namespaces=NAMESPACE)


def test_metadata():
    source = document(text_region())
    source.get_Page().add_AlternativeImage(AlternativeImageType(filename='binarized.png'))
    source.get_Page().set_orientation(math.nan)
    tree = build(source, LayoutOptions(right_to_left=True), 'Müller\x00 & Söhne.png')
    page = find(tree, '//pc:Page')[0]
    assert find(tree, '//pc:Creator/text()') == [CREATOR]
    assert CREATOR == 'eynollah 0.10.1'
    assert page.get('imageFilename') == 'Müller & Söhne.png'
    assert (page.get('imageWidth'), page.get('imageHeight')) == ('100', '100')
    assert page.get('readingDirection') == 'right-to-left'
    assert page.get('orientation') is None
    assert not find(tree, '//pc:AlternativeImage')


def test_invalid_confidence_is_removed():
    tree = build(document(text_region(conf=math.nan), text_region('region_0002', conf=0.5)))
    assert [coords.get('conf') for coords in find(tree, '//pc:TextRegion/pc:Coords')] == [None, '0.5']


def test_degenerate_elements_are_removed():
    region = text_region(lines=(SQUARE, '10,10 20,10 30,10', '5,5 5,5'))
    tree = build(document(region, text_region('region_0002', points='1,1 2,2'), border='0,0 1,1'))
    assert find(tree, '//pc:TextRegion/@id') == ['region_0001']
    assert len(find(tree, '//pc:TextLine')) == 1
    assert not find(tree, '//pc:Border')


def test_reading_order_only_references_existing_regions():
    source = document(text_region('region_0001', points='1,1 2,2'), text_region('region_0002'))
    xml_reading_order(source.get_Page(), [0, 1, 4])
    tree = build(source)
    assert find(tree, '//pc:RegionRefIndexed/@regionRef') == ['region_0002']
    assert find(tree, '//pc:RegionRefIndexed/@index') == ['0']


def test_reading_order_without_references_is_removed():
    source = document(text_region())
    xml_reading_order(source.get_Page(), [7])
    assert not find(build(source), '//pc:ReadingOrder')


def test_page_without_regions():
    assert build(document()) is None
    assert build(document(text_region(points='1,1 2,2'))) is None


def test_image_regions_are_kept():
    tree = build(document(ImageRegionType(id='region_0001', Coords=CoordsType(points=SQUARE))))
    assert find(tree, '//pc:ImageRegion/@id') == ['region_0001']


def test_lines_mode_uses_the_page_outline():
    options = LayoutOptions(lines_only=True)
    tree = build(document(text_region(points='50,50 99,50 99,99'), border='20,20 60,20 60,60'), options)
    assert find(tree, '//pc:TextRegion/pc:Coords/@points') == find(tree, '//pc:Border/pc:Coords/@points')


def test_lines_mode_without_lines():
    assert build(document(text_region(lines=())), LayoutOptions(lines_only=True)) is None


def test_polygons_are_simplified():
    stairs = [(x, x % 2) for x in range(100)] + [(99, 50), (0, 50)]
    assert len(clean_polygon(stairs, 100, 100, 1)) < 10
    assert len(clean_polygon(stairs, 100, 100, 0)) == len(stairs)


def test_polygons_are_limited_to_the_image():
    polygon = clean_polygon([(-5, -5), (120, -5), (120, 120), (-5, 120)], 100, 80, 1)
    assert polygon == [(0, 0), (99, 0), (99, 79), (0, 79)]


def test_spikes_are_removed_from_polygons():
    polygon = clean_polygon([(10, 10), (90, 10), (90, 90), (50, 90), (50, 99), (50, 90), (10, 90)], 100, 100, 0)
    assert Polygon(polygon).is_valid
    assert Polygon(polygon).area == 6400


@pytest.mark.parametrize('points', [[], [(1, 1)], [(1, 1), (1, 1), (5, 5)], [(0, 0), (5, 5), (9, 9)]])
def test_degenerate_polygons(points):
    assert clean_polygon(points, 100, 100, 1) is None
