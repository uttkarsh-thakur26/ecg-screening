"""Load one PTB-XL record at 100 Hz and plot one lead against time in seconds.

    python -m src.data.inspect_record              # first training-fold record (lowest ecg_id)
    python -m src.data.inspect_record --ecg-id 42

The lead, output path and ECG-paper grid come from the sample_plot section of the config.
A training-fold record is the default so this inspection never looks at validation or test data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path  # noqa: E402
from src.data.records import load_record, read_database  # noqa: E402


def pick_record(db: pd.DataFrame, train_folds: list[int], ecg_id: int | None) -> int:
    if ecg_id is not None:
        if ecg_id not in db.index:
            raise KeyError(f"ecg_id {ecg_id} is not in the database")
        return ecg_id
    train = db[db["strat_fold"].isin(train_folds)]
    if train.empty:
        raise ValueError(f"No records in training folds {train_folds}")
    return int(train.index.min())


def plot_lead(signal: np.ndarray, fields: dict[str, Any], lead: str, plot_cfg: dict[str, Any], title: str, out: Path) -> None:
    names = list(fields["sig_name"])
    if lead not in names:
        raise ValueError(f"Lead {lead!r} not in record; leads are {names}")
    i = names.index(lead)
    if fields["units"][i] != "mV":
        raise ValueError(f"Lead {lead} is in {fields['units'][i]!r}; the ECG grid assumes mV")

    fs = fields["fs"]
    t = np.arange(signal.shape[0]) / fs
    y = signal[:, i]
    major_mv = plot_cfg["grid_major_mv"]

    fig, ax = plt.subplots(figsize=(plot_cfg["figure_width_in"], plot_cfg["figure_width_in"] / 4))
    ax.plot(t, y, color="black", linewidth=0.8)
    ax.set_xlim(0, signal.shape[0] / fs)
    ax.set_ylim(np.floor(y.min() / major_mv) * major_mv, np.ceil(y.max() / major_mv) * major_mv)
    ax.xaxis.set_major_locator(MultipleLocator(plot_cfg["grid_major_s"]))
    ax.xaxis.set_minor_locator(MultipleLocator(plot_cfg["grid_minor_s"]))
    ax.yaxis.set_major_locator(MultipleLocator(major_mv))
    ax.yaxis.set_minor_locator(MultipleLocator(plot_cfg["grid_minor_mv"]))
    # Gridlines every 0.2 s, but label whole seconds only.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if np.isclose(v, round(v)) else ""))
    ax.grid(which="major", color="#e3a3a3", linewidth=0.6)
    ax.grid(which="minor", color="#f4d9d9", linewidth=0.3)
    # 25 mm/s and 10 mm/mV: one second is drawn 2.5x as wide as one millivolt is tall.
    ax.set_aspect(plot_cfg["gain_mm_per_mv"] / plot_cfg["paper_speed_mm_per_s"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Lead {lead} (mV)")
    ax.set_title(title, fontsize=10)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=plot_cfg["dpi"], bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load one PTB-XL record and plot one lead.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--ecg-id", type=int, default=None, help="default: lowest ecg_id in the training folds")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ds, plot_cfg = cfg["dataset"], cfg["sample_plot"]
    raw_dir = project_path(cfg["paths"]["raw_dir"])

    db = read_database(raw_dir, ds["database_csv"])
    ecg_id = pick_record(db, cfg["splits"]["train_folds"], args.ecg_id)
    row = db.loc[ecg_id]
    rel = row[ds["record_path_column"]]
    signal, fields = load_record(raw_dir, rel, ds)

    print(f"ecg_id {ecg_id} (strat_fold {int(row['strat_fold'])}): {rel}")
    print(f"signal shape: {signal.shape}  (samples, leads)")
    print(f"sampling rate: {fields['fs']} Hz, duration {signal.shape[0] / fields['fs']:g} s")
    print(f"leads: {', '.join(fields['sig_name'])}")
    print(f"units: {sorted(set(fields['units']))}")

    lead = plot_cfg["lead"]
    out = project_path(plot_cfg["output"])
    title = f"PTB-XL v{ds['version']}  |  ecg_id {ecg_id}  |  lead {lead}  |  {fields['fs']} Hz"
    plot_lead(signal, fields, lead, plot_cfg, title, out)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
