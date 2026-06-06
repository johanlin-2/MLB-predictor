"""Isotonic recalibration, reliability diagrams, Brier gate (per-model).

Per the approved plan the gate is **per-model**: if the moneyline model is
well-calibrated but the runline model is broken, only the runline picks are
suppressed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")          # headless safety for cron / CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

import config
from models._common import save_model

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    model_name: str
    brier_raw: float
    brier_calibrated: float
    passed_gate: bool
    plot_path: str


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels)
    return iso


def reliability_plot(probs: np.ndarray, labels: np.ndarray,
                     title: str, out_path) -> None:
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_and_calibrate(model_name: str,
                           val_probs: np.ndarray,
                           val_labels: np.ndarray) -> CalibrationReport:
    brier_raw = float(brier_score_loss(val_labels, val_probs))
    iso = fit_isotonic(val_probs, val_labels)
    cal = iso.transform(val_probs)
    brier_cal = float(brier_score_loss(val_labels, cal))
    passed = brier_cal <= config.BRIER_GATE

    plot_path = config.CALIBRATION_DIR / f"{model_name}_reliability.png"
    reliability_plot(cal, val_labels, f"{model_name} (calibrated)", plot_path)

    save_model(iso, f"{model_name}_isotonic")

    report = CalibrationReport(
        model_name=model_name, brier_raw=brier_raw, brier_calibrated=brier_cal,
        passed_gate=passed, plot_path=str(plot_path),
    )
    logger.info("%s: Brier raw=%.3f calibrated=%.3f gate=%s",
                model_name, brier_raw, brier_cal, "PASS" if passed else "FAIL")
    return report


def apply_calibration(model_name: str, probs: np.ndarray) -> np.ndarray:
    from models._common import load_model
    iso: IsotonicRegression = load_model(f"{model_name}_isotonic")
    return iso.transform(probs)


def write_summary(reports: list[CalibrationReport]) -> None:
    path = config.CALIBRATION_DIR / "summary.json"
    with open(path, "w") as f:
        json.dump([r.__dict__ for r in reports], f, indent=2)
    logger.info("wrote calibration summary to %s", path)
