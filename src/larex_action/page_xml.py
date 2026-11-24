# SPDX-License-Identifier: MIT
import math
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib.metadata import version

from ocrd_models.ocrd_page import PcGtsType, to_xml
from shapely import Polygon, make_valid
from shapely.geometry.base import BaseGeometry

from .parameters import LayoutOptions

CREATOR = f'eynollah {version("eynollah")}'
REGION_SETTERS = ('TextRegion', 'TableRegion', 'ImageRegion', 'SeparatorRegion')
MIN_REPAIRED_SHARE = 0.9
INVALID_XML_CHARACTERS = re.compile('[^\t\n\r -퟿-�\U00010000-\U0010ffff]')

Point = tuple[float, float]


@dataclass(frozen=True)
class PageMetadata:
    file_name: str
    width: int
    height: int


def finalize(document: PcGtsType, metadata: PageMetadata, options: LayoutOptions) -> bytes | None:
    page = document.get_Page()
    _set_metadata(document, metadata, options)
    if options.lines_only:
        _restore_page_outline(page)
    _clean_border(page, metadata, options.polygon_tolerance)
    for setter in REGION_SETTERS:
        regions = getattr(page, f'get_{setter}')()
        kept = [region for region in regions if _clean(region, metadata, options.polygon_tolerance)]
        getattr(page, f'set_{setter}')(kept)
    if options.lines_only:
        page.set_TextRegion([region for region in page.get_TextRegion() if region.get_TextLine()])
    regions = list(_regions(page))
    if not regions:
        return None
    _clean_reading_order(page, {region.get_id() for region in regions})
    return to_xml(document).encode('utf-8')


def _regions(page) -> Iterator:
    for setter in REGION_SETTERS:
        yield from getattr(page, f'get_{setter}')()


def _set_metadata(document: PcGtsType, metadata: PageMetadata, options: LayoutOptions) -> None:
    document.get_Metadata().set_Creator(CREATOR)
    page = document.get_Page()
    page.set_imageFilename(INVALID_XML_CHARACTERS.sub('', metadata.file_name) or 'page')
    page.set_imageWidth(metadata.width)
    page.set_imageHeight(metadata.height)
    page.set_readingDirection('right-to-left' if options.right_to_left else 'left-to-right')
    page.set_AlternativeImage([])
    if not _is_finite(page.get_orientation()):
        page.set_orientation(None)


def _restore_page_outline(page) -> None:
    border = page.get_Border()
    for region in page.get_TextRegion():
        region.get_Coords().set_points(border.get_Coords().get_points())


def _clean_border(page, metadata: PageMetadata, tolerance: float) -> None:
    border = page.get_Border()
    if border is not None and not _clean_coords(border.get_Coords(), metadata, tolerance):
        page.set_Border(None)


def _clean(element, metadata: PageMetadata, tolerance: float) -> bool:
    if not _clean_coords(element.get_Coords(), metadata, tolerance):
        return False
    if hasattr(element, 'get_orientation') and not _is_finite(element.get_orientation()):
        element.set_orientation(None)
    if hasattr(element, 'get_TextLine'):
        element.set_TextLine([line for line in element.get_TextLine() if _clean(line, metadata, tolerance)])
    return True


def _clean_coords(coords, metadata: PageMetadata, tolerance: float) -> bool:
    if not _is_finite(coords.get_conf()):
        coords.set_conf(None)
    points = clean_polygon(parse_points(coords.get_points()), metadata.width, metadata.height, tolerance)
    if points is None:
        return False
    coords.set_points(' '.join(f'{x},{y}' for x, y in points))
    return True


def _clean_reading_order(page, region_ids: set[str]) -> None:
    order = page.get_ReadingOrder()
    if order is None:
        return
    group = order.get_OrderedGroup()
    references = [ref for ref in group.get_RegionRefIndexed() if ref.get_regionRef() in region_ids]
    if not references:
        page.set_ReadingOrder(None)
        return
    for index, reference in enumerate(references):
        reference.set_index(index)
    group.set_id('reading_order')
    group.set_RegionRefIndexed(references)


def parse_points(value: str) -> list[Point]:
    return [(float(x), float(y)) for x, y in (pair.split(',') for pair in value.split())]


def clean_polygon(points: list[Point], width: int, height: int, tolerance: float) -> list[tuple[int, int]] | None:
    outline = _outline([(min(max(x, 0), width - 1), min(max(y, 0), height - 1)) for x, y in points])
    if outline is None:
        return None
    polygon = Polygon(outline)
    if tolerance:
        simplified = polygon.simplify(tolerance)
        if isinstance(simplified, Polygon) and not simplified.is_empty:
            polygon = simplified
    outline = _outline(polygon.exterior.coords)
    if outline is None:
        return None
    polygon = Polygon(outline)
    if polygon.is_valid:
        return outline
    return _repaired(polygon) or outline


def _repaired(polygon: Polygon) -> list[tuple[int, int]] | None:
    parts = [part for part in _polygons(make_valid(polygon)) if part.area]
    if not parts:
        return None
    largest = max(parts, key=lambda part: part.area)
    if largest.area < MIN_REPAIRED_SHARE * sum(part.area for part in parts):
        return None
    return _outline(largest.exterior.coords)


def _polygons(geometry: BaseGeometry) -> Iterator[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    for part in getattr(geometry, 'geoms', ()):
        yield from _polygons(part)


def _outline(points: Iterable[Point]) -> list[tuple[int, int]] | None:
    outline = _without_repetitions([(round(x), round(y)) for x, y in points])
    if len(outline) < 3 or not _spans_an_area(outline):
        return None
    return outline


def _spans_an_area(outline: list[tuple[int, int]]) -> bool:
    (x0, y0), *rest = outline
    x1, y1 = rest[0]
    return any((x1 - x0) * (y - y0) != (y1 - y0) * (x - x0) for x, y in rest[1:])


def _without_repetitions(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    unique = [point for index, point in enumerate(points) if index == 0 or point != points[index - 1]]
    if len(unique) > 1 and unique[0] == unique[-1]:
        unique.pop()
    return unique


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)
