# SPDX-License-Identifier: MIT
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from eynollah.eynollah import Eynollah
from eynollah.model_zoo import EynollahModelZoo
from eynollah.model_zoo.model_zoo import MODEL_VRAM_LIMITS
from eynollah.utils.resize import resize_image
from eynollah.utils.xml import create_page_xml
from PIL import Image

from .page_xml import PageMetadata, finalize
from .parameters import LayoutOptions

ORT_LOG_SEVERITY = 3
UNUSED_OUTPUT_DIRECTORY = '/nonexistent'

logger = logging.getLogger(__name__)


class DeviceUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceSettings:
    device: str = 'auto'
    threads: int = 1
    memory_limits: bool = False

    @property
    def device_id(self) -> int:
        return int(self.device.partition(':')[2] or 0)


def resolve_device(device: str) -> str:
    if device == 'cpu':
        return 'cpu'
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        if device == 'auto':
            return 'cpu'
        raise DeviceUnavailable('CUDA was requested, but this image only contains the CPU runtime')
    if device == 'auto' and not _has_nvidia_device():
        return 'cpu'
    return 'cuda' if device == 'auto' else device


def _has_nvidia_device() -> bool:
    return any(Path('/dev').glob('nvidia[0-9]*')) or Path('/dev/dxg').exists()


class OnnxModel:
    def __init__(self, name: str, path: Path, settings: DeviceSettings):
        self.name = name
        self._lock = threading.Lock()
        options, providers = _session_configuration(name, settings)
        self._session = ort.InferenceSession(str(path), options, providers=providers)
        active = self._session.get_providers()[0]
        if settings.device != 'cpu' and active != 'CUDAExecutionProvider':
            raise DeviceUnavailable(f'CUDA could not be initialized for the {name} model')
        self._input = self._session.get_inputs()[0]
        self.input_shape = self._input.shape
        self._output = self._session.get_outputs()[0].name

    def predict(self, data: np.ndarray, verbose: int = 0) -> np.ndarray:
        batch = np.asarray(data, dtype=np.float32)
        with self._lock:
            return self._session.run([self._output], {self._input.name: batch})[0]

    __call__ = predict


def _session_configuration(name: str, settings: DeviceSettings) -> tuple[ort.SessionOptions, list]:
    options = ort.SessionOptions()
    options.log_severity_level = ORT_LOG_SEVERITY
    if settings.device == 'cpu':
        options.intra_op_num_threads = settings.threads
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        return options, ['CPUExecutionProvider']
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.add_session_config_entry('session.inter_op.allow_spinning', '0')
    cuda = {'device_id': settings.device_id, 'arena_extend_strategy': 'kSameAsRequested'}
    if settings.memory_limits:
        cuda['gpu_mem_limit'] = MODEL_VRAM_LIMITS[name] * 1024 * 1024
    return options, [('CUDAExecutionProvider', cuda), 'CPUExecutionProvider']


class ModelZoo(EynollahModelZoo):
    def __init__(self, directory: Path, settings: DeviceSettings):
        super().__init__(str(directory))
        self._settings = settings

    def load_models(self, *names: str, device: str = '') -> dict:
        for name in names:
            if name not in self._loaded:
                self._loaded[name] = OnnxModel(name, self.model_path(name), self._settings)
        return self._loaded

    def shutdown(self) -> None:
        return None


class LineAnalyzer(Eynollah):
    def __init__(self, **options):
        self._state = threading.local()
        super().__init__(skip_layout_and_reading_order=True, **options)

    def run_textline(self, image_page):
        mask, self._state.confidence = super().run_textline(image_page)
        return mask, self._state.confidence

    def get_early_layout(self, *args, **kwargs):
        *layout, mask, regions_confidence, _ = super().get_early_layout(*args, **kwargs)
        confidence = resize_image(self._state.confidence, *mask.shape[:2])
        return *layout, mask, regions_confidence, confidence


class LayoutEngine:
    def __init__(self, directory: Path, settings: DeviceSettings):
        ort.set_default_logger_severity(ORT_LOG_SEVERITY)
        if settings.device != 'cpu':
            ort.preload_dlls()
        self._zoo = ModelZoo(directory, settings)
        self._analyzers: dict[LayoutOptions, Eynollah] = {}
        self._lock = threading.Lock()

    def load(self, options: LayoutOptions) -> Eynollah:
        with self._lock:
            if options not in self._analyzers:
                self._analyzers.clear()
                self._analyzers[options] = self._create(options)
            return self._analyzers[options]

    def process(self, image: Image.Image, file_name: str, options: LayoutOptions) -> bytes | None:
        document = create_page_xml(file_name, image.height, image.width)
        self.load(options).run_single(
            file_name,
            dir_out=UNUSED_OUTPUT_DIRECTORY,
            img_pil=image,
            pcgts=document,
        )
        return finalize(document, PageMetadata(file_name, image.width, image.height), options)

    def _create(self, options: LayoutOptions) -> Eynollah:
        analyzer = LineAnalyzer if options.lines_only else Eynollah
        return analyzer(
            model_zoo=self._zoo,
            full_layout=options.full_layout,
            tables=options.tables,
            curved_line=options.curved_lines,
            right2left=options.right_to_left,
            reading_order_machine_based=options.model_reading_order,
            ignore_page_extraction=not options.crop_page,
            input_binary=options.binarize,
            num_col_lower=options.min_columns,
            num_col_upper=options.max_columns,
            allow_marginalia=options.marginalia,
            paragraphs_from_marginalia=options.split_paragraphs_at_marginalia,
            threshold_art_class_layout=options.region_separation,
            threshold_art_class_textline=options.line_separation,
            logger=logging.getLogger('eynollah'),
        )
