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
results
```

Usage:

```bash
python download_imagenet_subset.py
python compute_metrics.py
```
