# CS 534 Final Project

Files:

- `download_imagenet_subset.py`: download the selected ImageNet classes
- `compute_metrics.py`: compute metrics for original vs generated images
- `config.yaml`: paths and category settings
- `requirements.txt`: dependencies
- `io_utils.py`, `image_registry.py`, `metrics_engine.py`: helpers used by `compute_metrics.py`

Expected folders:

```text
data/inputs
data/generated
```

`data/metadata.csv` is generated automatically by `compute_metrics.py` from
the image files under `data/inputs` and `data/generated`. It is intentionally
not committed because it contains machine-specific absolute file paths.

Usage:

```bash
python download_imagenet_subset.py
python compute_metrics.py
```
