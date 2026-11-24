# SPDX-License-Identifier: MIT
import numpy as np
import pytest
from PIL import Image

from larex_action.images import UnreadableImage, load_rgb


def save(tmp_path, image, name='page.png', **options):
    path = tmp_path / name
    image.save(path, **options)
    return path


def pixel(path):
    image = load_rgb(path, 1_000_000)
    assert image.mode == 'RGB'
    return image.getpixel((0, 0))


@pytest.mark.parametrize(
    ('image', 'expected'),
    [
        (Image.new('1', (4, 4), 1), (255, 255, 255)),
        (Image.new('L', (4, 4), 90), (90, 90, 90)),
        (Image.new('RGBA', (4, 4), (200, 10, 10, 0)), (255, 255, 255)),
        (Image.new('LA', (4, 4), (0, 255)), (0, 0, 0)),
        (Image.fromarray(np.full((4, 4), 65535, np.uint16)), (255, 255, 255)),
        (Image.fromarray(np.full((4, 4), 25700, np.uint16)), (100, 100, 100)),
    ],
)
def test_color_modes(tmp_path, image, expected):
    assert pixel(save(tmp_path, image)) == expected


def test_float_image(tmp_path):
    image = Image.fromarray(np.full((4, 4), 0.5, np.float32))
    assert pixel(save(tmp_path, image, 'page.tif')) == (128, 128, 128)


def test_cmyk_image(tmp_path):
    assert pixel(save(tmp_path, Image.new('CMYK', (4, 4), (0, 0, 0, 0)), 'page.tif')) == (255, 255, 255)


def test_transparent_palette(tmp_path):
    image = Image.new('P', (4, 4), 0)
    image.putpalette([10, 20, 30])
    assert pixel(save(tmp_path, image, transparency=0)) == (255, 255, 255)


def test_exif_orientation_is_ignored(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6
    path = save(tmp_path, Image.new('RGB', (8, 4)), 'page.jpg', exif=exif)
    assert load_rgb(path, 1_000).size == (8, 4)


def test_first_frame_is_used(tmp_path):
    frames = [Image.new('RGB', (4, 4), color) for color in ('red', 'blue')]
    path = save(tmp_path, frames[0], 'page.tif', save_all=True, append_images=frames[1:])
    assert pixel(path) == (255, 0, 0)


def test_pixel_limit(tmp_path):
    with pytest.raises(UnreadableImage, match='more than the limit of 100'):
        load_rgb(save(tmp_path, Image.new('L', (11, 10))), 100)


@pytest.mark.parametrize('size', [(10, 101), (101, 10)])
def test_extreme_proportions(tmp_path, size):
    with pytest.raises(UnreadableImage, match='at most 10 times as long'):
        load_rgb(save(tmp_path, Image.new('L', size)), 10_000)


def test_unsupported_file(tmp_path):
    path = tmp_path / 'page.png'
    path.write_bytes(b'not an image')
    with pytest.raises(UnreadableImage, match='not supported'):
        load_rgb(path, 100)


def test_truncated_file(tmp_path):
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), np.uint8))
    path = save(tmp_path, noise)
    path.write_bytes(path.read_bytes()[:4000])
    with pytest.raises(UnreadableImage, match='could not be read'):
        load_rgb(path, 10_000)
