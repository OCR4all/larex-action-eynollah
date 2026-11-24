# LAREX Action: eynollah

Layout analysis for [LAREX](https://github.com/OCR4all/LAREX) with [eynollah](https://github.com/qurator-spk/eynollah) 0.10.1.
The Action detects regions, text lines, and the reading order of page images and returns PAGE XML.

Text recognition, image enhancement, standalone binarization, and standalone reading order detection are not included.

## Setup

1. Copy `template.env` to `.env`, then set the dispatch secret and the origin of the LAREX server.

   ```bash
   cp template.env .env
   ```

2. Create the model directory and download the models. The archive has 1.9 GB and is verified against a pinned checksum; about 3 GB of free space are needed during the download, 1.1 GB afterwards. The models are published by the Staatsbibliothek zu Berlin under [CC BY 4.0](https://zenodo.org/records/21381102).

   ```bash
   mkdir -p models
   MODEL_UID=$(id -u) MODEL_GID=$(id -g) docker compose run --rm --build models
   ```

   The container only needs read access to the models afterwards.

3. Start the processor.

   ```bash
   docker compose up -d --build
   ```

   For NVIDIA GPUs, add the overlay. It needs an x86_64 host with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and its CDI configuration.

   ```bash
   docker compose -f compose.yml -f compose.gpu.yml up -d --build
   ```

4. Adjust the URLs and `secretRef` in `action.yml` and add the definition in LAREX. The endpoint path has to start with one of `LAREX_ROUTE_PREFIXES`.

LAREX has to support incremental page results and run reports.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `mode` | `layout` | `layout` analyses the whole page. `lines` only detects text lines and places them in one region with the outline of the page. |
| `fullLayout` | `true` | Distinguish headings and drop capitals from paragraphs. |
| `tables` | `false` | Detect table regions. |
| `marginalia` | `both` | `both`, `left`, `right`, or `off`. Applies to pages with one or two columns. |
| `splitParagraphsAtMarginalia` | `false` | Split the main text at the height of neighbouring marginalia. |
| `readingOrder` | `heuristic` | `heuristic` or the slower `model`. |
| `rightToLeft` | `false` | Read the columns from right to left. |
| `curvedLines` | `false` | Detect the lines of each region separately. Slower. |
| `cropPage` | `true` | Restrict the analysis to the detected page frame. |
| `binarize` | `false` | Binarize the image before the analysis. The stored image is not changed. |
| `minColumns`, `maxColumns` | `0` | Limits for the number of columns (1 to 6), `0` for no limit. |
| `regionSeparation`, `lineSeparation` | `0.1` | Confidence thresholds for boundaries between regions and lines. |
| `polygonTolerance` | `1` | Maximum deviation in pixels when polygons are simplified, `0` keeps every point. |

In `lines` mode only `cropPage`, `binarize`, `rightToLeft`, the column limits, `lineSeparation`, and `polygonTolerance` apply.

## Results

- The result replaces the existing annotations of a page. LAREX keeps the previous version.
- Pages on which nothing is detected are reported as skipped and keep their annotations.
- A page fails without affecting the others when its image is missing, unreadable, larger than `EYNOLLAH_MAX_IMAGE_PIXELS`, or more than ten times as long as wide.
- Coordinates refer to the stored pixels. EXIF orientation is ignored, as in LAREX.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `LAREX_PROCESSOR_ID` | | Has to match `id` in `action.yml`. |
| `LAREX_DISPATCH_HMAC_SECRET` | | Secret behind `secretRef`. |
| `LAREX_ALLOWED_CALLBACK_ORIGINS` | | JSON list with the origins of the LAREX server. |
| `LAREX_ROUTE_PREFIXES` | `[]` | JSON list of path prefixes. |
| `LAREX_SDK_TRANSPORT_DEBUG` | `false` | Log the requests to LAREX. |
| `ACTION_DEBUG` | `false` | Verbose logs, including those of eynollah. |
| `ACTION_MAX_CONCURRENT_RUNS` | `1` | Runs that are processed at the same time. Further runs wait. Every concurrent run uses its own worker process and its own copy of the models. |
| `ACTION_LOADING_TIMEOUT` | `300` | Seconds until loading the models fails the run, `None` for no limit. |
| `ACTION_PROCESSING_TIMEOUT` | `900` | Seconds per page, `None` for no limit. |
| `ACTION_TIMEOUT_MODE` | `fail_page` | `fail_page` continues with the next page, `fail_action` stops the run. |
| `ACTION_IDLE_TIMEOUT` | `600` | Seconds for which a worker keeps its models in memory after a run, `0` to release them immediately. |
| `EYNOLLAH_MODEL_DIR` | `/models` | Directory that contains `models_eynollah`. |
| `EYNOLLAH_DEVICE` | `auto` | `auto`, `cpu`, `cuda`, or `cuda:N`. A run fails if CUDA was selected, or a GPU is visible, but cannot be used. |
| `EYNOLLAH_PAGE_WORKERS` | automatic | Pages that are processed at the same time: 1 on the CPU, up to 4 on a GPU. |
| `EYNOLLAH_CPU_THREADS` | automatic | Threads per worker. By default the CPUs available to the container, divided by the number of concurrent runs. |
| `EYNOLLAH_GPU_MEMORY_LIMITS` | `false` | Limit the GPU memory of every model to the values used by eynollah. |
| `EYNOLLAH_MAX_IMAGE_PIXELS` | `300000000` | Largest accepted image. |

## How it works

- The web process holds no models. Every run uses a separate worker process that loads the models with ONNX Runtime and stays warm for following runs.
- Pages are downloaded ahead and submitted as soon as they are finished.
- eynollah starts three additional processes per model and copies every batch between them. The Action runs the models in the worker process instead; the analysis itself is unchanged.
- On the CPU, inference uses all threads, so more than one page at a time does not help. On a GPU several pages share the models, so that the preparation of the pages does not leave the GPU waiting.
- Cancellation, timeouts, and crashes stop the worker process. Pages that were processed next to a crashed page are repeated on their own, so only the responsible page fails. Three such failures in a row stop the run.
- When the processor is stopped, active runs are reported as failed.

### Differences from eynollah

- In eynollah 0.10.1, text line only mode (`--skip_layout_and_reading_order`) stops with an error because the line confidences are missing, and shifts the outline of the region. Both are corrected here.
- A single column limit fixes the number of columns in eynollah. Here it only limits it.
- `--headers_off`, `--allow_scaling`, and `--allow_enhancement` are not available. The first assigns the reading order to wrong regions, the others have no effect or need a model that is not loaded.

### Pinned versions

- `onnxruntime-gpu` 1.26.0 is the last release for CUDA 12.
- The dependency metadata of eynollah is overridden in `pyproject.toml`, so that its GPU and TensorRT dependencies are only installed in the GPU image. TensorRT is not included.

## Development

```bash
uv sync --extra cpu --extra test
```

```bash
uv run --no-sync pytest
```

Tests with the published models:

```bash
EYNOLLAH_MODEL_DIR=./models uv run --no-sync pytest -m models
```

## License

MIT. eynollah is licensed under Apache 2.0, its models under CC BY 4.0.
