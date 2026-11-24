# SPDX-License-Identifier: MIT
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

MAX_ASPECT_RATIO = 10


class UnreadableImage(ValueError):
    pass


def load_rgb(path: Path, max_pixels: int) -> Image.Image:
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            if image.width * image.height > max_pixels:
                raise UnreadableImage(
                    f'The image has {image.width}x{image.height} pixels, more than the limit of {max_pixels}'
                )
            if max(image.size) > MAX_ASPECT_RATIO * min(image.size):
                raise UnreadableImage(
                    f'The image has {image.width}x{image.height} pixels, '
                    f'but one side may be at most {MAX_ASPECT_RATIO} times as long as the other'
                )
            image.load()
            return _to_rgb(image)
    except UnidentifiedImageError as error:
        raise UnreadableImage('The image format is not supported') from error
    except (OSError, SyntaxError) as error:
        raise UnreadableImage(f'The image could not be read: {error}') from error


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode.startswith(('I', 'F')):
        return _scale_to_8_bit(image).convert('RGB')
    if image.has_transparency_data:
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert('RGB')
    return image.convert('RGB')


def _scale_to_8_bit(image: Image.Image) -> Image.Image:
    values = np.asarray(image).astype(np.float32)
    if image.mode.startswith('I;16') or (image.mode == 'I' and values.max(initial=0) > 255):
        values = values / 257
    elif image.mode == 'F' and values.max(initial=0) <= 1:
        values = values * 255
    return Image.fromarray(np.clip(np.rint(values), 0, 255).astype(np.uint8))
