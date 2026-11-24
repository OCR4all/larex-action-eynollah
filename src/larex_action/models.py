# SPDX-License-Identifier: MIT
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from .parameters import LayoutOptions

ARCHIVE_URL = 'https://zenodo.org/api/records/21381102/files/models_inference_layout_v0_9_1.zip/content'
ARCHIVE_SHA256 = 'd12fc7ca19fb30800314e78a17005af19059af5620019712ae4565f93687408a'
ARCHIVE_SIZE = 1_847_700_967
DOWNLOAD_ATTEMPTS = 3
CHUNK_SIZE = 1 << 20


@dataclass(frozen=True)
class ModelFile:
    path: str
    size: int
    sha256: str


MODEL_FILES = {
    'binarization': ModelFile(
        'models_eynollah/eynollah-binarization_20210425.onnx',
        152_434_670,
        'f86248e7ae1e05ef18a48c9adfd06daacf77ec69ec4ae359634986babc2470b4',
    ),
    'col_classifier': ModelFile(
        'models_eynollah/eynollah-column-classifier_20210425.onnx',
        102_360_983,
        'c9b1ed9fcd608880e2c750c13e557d33a201c3f06436dd9da886d1557ce20c0d',
    ),
    'page': ModelFile(
        'models_eynollah/model_eynollah_page_extraction_20250915.onnx',
        152_433_962,
        '419cf02957258bc38b917776bb9649ef9a8c1e89c6a135aa63b8e3965103e8d5',
    ),
    'reading_order': ModelFile(
        'models_eynollah/model_eynollah_reading_order_20250824.onnx',
        106_813_881,
        '0383062e519fa1eba5e5b9fe2b404c6dd28abaf5e7394b216197b53033afdad2',
    ),
    'region_1_2': ModelFile(
        'models_eynollah/modelens_e_l_all_sp_0_1_2_3_4_171024.onnx',
        149_848_856,
        'cba9bf1b53dd53b0c537496bf4b834a9edf1f374a117cbc974df2ff4ebf299a4',
    ),
    'region_fl_np': ModelFile(
        'models_eynollah/modelens_full_lay_1__4_3_091124.onnx',
        149_626_714,
        '4119c89e6dc0f0139120fa49f3518c03adcab6089081f1cbdcd3e4205f2384f3',
    ),
    'table': ModelFile(
        'models_eynollah/modelens_table_0t4_201124.onnx',
        149_626_096,
        'e98dc8f50e041ff92e74741311e4fa9820accca1d9877c84e2086f0ce16d3670',
    ),
    'textline': ModelFile(
        'models_eynollah/modelens_textline_0_1__2_4_16092024.onnx',
        149_735_395,
        'bc6575898c41bba852b3c5367c92add6663ba121e45c325fe770e1cc05241e5c',
    ),
}


class ModelDownloadError(RuntimeError):
    pass


def required_models(options: LayoutOptions) -> list[str]:
    names = ['col_classifier', 'page', 'textline', 'region_1_2']
    if options.binarize:
        names.append('binarization')
    if options.full_layout:
        names.append('region_fl_np')
    if options.model_reading_order:
        names.append('reading_order')
    if options.tables:
        names.append('table')
    return names


def missing_models(directory: Path, names: list[str]) -> list[str]:
    return [name for name in names if not (directory / MODEL_FILES[name].path).is_file()]


def verify_models(directory: Path) -> list[str]:
    problems = []
    for name, file in MODEL_FILES.items():
        path = directory / file.path
        if not path.is_file():
            problems.append(f'{name}: {path} is missing')
        elif path.stat().st_size != file.size or _sha256(path) != file.sha256:
            problems.append(f'{name}: {path} does not match the published checksum')
    return problems


def download_models(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    required_space = ARCHIVE_SIZE + sum(file.size for file in MODEL_FILES.values())
    if shutil.disk_usage(directory).free < required_space:
        raise ModelDownloadError(f'{directory} needs {required_space / 1e9:.1f} GB of free space')
    with tempfile.TemporaryDirectory(prefix='.download-', dir=directory) as temp:
        archive = Path(temp) / 'models.zip'
        _download_archive(archive)
        for name, file in MODEL_FILES.items():
            _extract(archive, name, file, directory)


def _download_archive(target: Path) -> None:
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            _stream_archive(target)
            return
        except (httpx.HTTPError, ModelDownloadError) as error:
            print(f'Attempt {attempt}/{DOWNLOAD_ATTEMPTS} failed: {error}', file=sys.stderr)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise ModelDownloadError(f'Could not download {ARCHIVE_URL}: {error}') from error


def _stream_archive(target: Path) -> None:
    digest = hashlib.sha256()
    received = 0
    reported = 0
    with (
        httpx.stream('GET', ARCHIVE_URL, follow_redirects=True, timeout=60) as response,
        target.open('wb') as output,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes(CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            percent = received * 100 // ARCHIVE_SIZE
            if percent >= reported + 10:
                reported = percent - percent % 10
                print(f'Downloaded {reported}%', flush=True)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ModelDownloadError(f'{ARCHIVE_URL} does not match the pinned checksum')


def _extract(archive: Path, name: str, file: ModelFile, directory: Path) -> None:
    target = directory / file.path
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(target.name)
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as source, source.open(file.path) as content, temporary.open('wb') as output:
        while chunk := content.read(CHUNK_SIZE):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != file.sha256:
        raise ModelDownloadError(f'{name} does not match the pinned checksum')
    temporary.replace(target)
    print(f'Installed {target}', flush=True)


def _sha256(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m larex_action.models')
    parser.add_argument('command', choices=['download', 'verify'])
    parser.add_argument('--directory', type=Path, default=Path(os.environ.get('EYNOLLAH_MODEL_DIR') or '/models'))
    parsed = parser.parse_args(arguments)
    problems = verify_models(parsed.directory)
    if parsed.command == 'download' and problems:
        try:
            download_models(parsed.directory)
        except (ModelDownloadError, OSError) as error:
            print(error, file=sys.stderr)
            return 1
        problems = verify_models(parsed.directory)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f'All models in {parsed.directory} are complete')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
