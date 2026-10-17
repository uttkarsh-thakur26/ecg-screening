"""Print one batch from each DataLoader and time one pass over the training loader.

    python -m src.data.inspect_pipeline                     # the configured dataloader.num_workers.train
    python -m src.data.inspect_pipeline --workers 0 2 4 8   # compare settings for the epoch time

Only data loading is timed (no model), with augmentation on, exactly as training will iterate it.
The test loader is opened with allow_test=True for one reason: to print its batch shapes. Nothing
about its contents (labels, values, statistics) is printed or computed.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Any

import torch

from src.config import DEFAULT_CONFIG_PATH, load_config
from src.data.cache import open_cache
from src.data.dataset import make_loader


def _describe_batch(name: str, loader: torch.utils.data.DataLoader, show_values: bool) -> None:
    ds = loader.dataset
    x, y = next(iter(loader))
    aug = ds.augmenter.describe() if ds.augment else "off"
    print(f"{name:<5} {len(ds):>6,} records, {len(loader):>3} batches of {loader.batch_size}, augmentation: {aug}")
    print(f"      signal {tuple(x.shape)} {str(x.dtype).removeprefix('torch.')}   "
          f"target {tuple(y.shape)} {str(y.dtype).removeprefix('torch.')}")
    if show_values:
        print(f"      first batch: signal mean {x.mean():+.3f}, std {x.std():.3f}; "
              f"positives per class {dict(zip(ds.classes, y.sum(0).int().tolist()))}")


def _time_epochs(cfg: dict[str, Any], workers: int, epochs: int) -> list[float]:
    cfg = copy.deepcopy(cfg)
    cfg["dataloader"]["num_workers"]["train"] = workers
    loader = make_loader(cfg, "train")
    times = []
    for _ in range(epochs):
        start = time.perf_counter()
        n = 0
        for x, _y in loader:
            n += x.shape[0]
        times.append(time.perf_counter() - start)
        if n != len(loader.dataset):
            raise RuntimeError(f"Epoch yielded {n} records, the dataset has {len(loader.dataset)}")
    return times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show one batch per loader and time a training epoch.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--workers", type=int, nargs="+", help="num_workers values to time (default: config)")
    parser.add_argument("--epochs", type=int, default=2, help="epochs to time per setting; the first includes start-up")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    start = time.perf_counter()
    cache = open_cache(cfg)
    print(f"Cache {cache.dir}: signals {tuple(cache.signals.shape)} {cache.signals.dtype}, "
          f"labels {tuple(cache.labels.shape)}, opened in {time.perf_counter() - start:.2f} s")
    print()

    _describe_batch("train", make_loader(cfg, "train"), show_values=True)
    _describe_batch("val", make_loader(cfg, "val"), show_values=True)
    _describe_batch("test", make_loader(cfg, "test", allow_test=True), show_values=False)  # shapes only
    try:
        make_loader(cfg, "test")
    except Exception as exc:  # noqa: BLE001 - printing the guard's own message
        print(f"      without allow_test: {type(exc).__name__}: {str(exc).split('.')[0]}.")
    else:
        raise RuntimeError("The test loader opened without allow_test=True; the guard is broken")
    print()

    batch = cfg["training"]["batch_size"]
    print(f"Time to load one training epoch (batch {batch}, augmentation on, no model):")
    for workers in args.workers or [cfg["dataloader"]["num_workers"]["train"]]:
        times = _time_epochs(cfg, workers, args.epochs)
        per_epoch = ", ".join(f"epoch {i + 1}: {t:.2f} s" for i, t in enumerate(times))
        rate = len(cache.rows("train")) / min(times)
        print(f"  num_workers={workers}: {per_epoch}  ({rate:,.0f} records/s at best)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
