# -*- coding: utf-8 -*-
"""Render report SVGs from grouped MNIST experiment CSV.

This script avoids Matplotlib text rendering so SVG labels stay readable on GitHub.
"""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "experiment_logs" / "grouped_strategy_run_20260527-115521.csv"
OUT_DIR = ROOT / "report_assets"

COLORS = {
    "adam_baseline": "#2563eb",
    "sgd_lr_0_01": "#dc2626",
    "no_batchnorm": "#16a34a",
    "no_dropout": "#9333ea",
    "adam_lr_0_01": "#ea580c",
    "adam_lr_decay": "#0891b2",
}

GROUPS = {
    "optimizer_sgd_vs_adam": ["sgd_lr_0_01", "adam_baseline"],
    "regularization_bn_dropout": ["adam_baseline", "no_batchnorm", "no_dropout"],
    "learning_rate_comparison": ["adam_baseline", "adam_lr_0_01", "adam_lr_decay"],
}

TITLES = {
    "optimizer_sgd_vs_adam": "SGD vs Adam",
    "regularization_bn_dropout": "BatchNorm and Dropout",
    "learning_rate_comparison": "Learning rate comparison",
}


def load_rows():
    grouped = defaultdict(list)
    with CSV_PATH.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            row["epoch"] = int(row["epoch"])
            row["lr"] = float(row["lr"])
            row["train_loss"] = float(row["train_loss"])
            row["train_acc"] = float(row["train_acc"])
            row["val_loss"] = float(row["val_loss"])
            row["val_acc"] = float(row["val_acc"])
            row["params"] = int(row["params"])
            grouped[row["strategy"]].append(row)
    return grouped


def scale(value, source_min, source_max, target_min, target_max):
    if source_min == source_max:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def padded_range(values, lower=None, upper=None, pad_ratio=0.12):
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0:
        span = max(abs(hi) * 0.1, 1e-3)
    lo -= span * pad_ratio
    hi += span * pad_ratio
    if lower is not None:
        lo = max(lower, lo)
    if upper is not None:
        hi = min(upper, hi)
    return lo, hi


def svg_start(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>',
        'text{font-family:Arial,Helvetica,sans-serif;fill:#111827}',
        '.title{font-size:20px;font-weight:700}',
        '.label{font-size:12px;fill:#4b5563}',
        '.small{font-size:11px;fill:#6b7280}',
        '.axis{stroke:#374151;stroke-width:1.2}',
        '.grid{stroke:#e5e7eb;stroke-width:1}',
        '</style>',
    ]


def write_svg(filename, lines):
    (OUT_DIR / filename).write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def draw_axes(lines, x, y, w, h, title, y_label, y_min, y_max, log_y=False):
    lines.append(f'<text x="{x}" y="30" class="title">{html.escape(title)}</text>')
    lines.append(f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" class="axis"/>')
    lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" class="axis"/>')

    ticks = 5
    for i in range(ticks):
        value = y_min + (y_max - y_min) * i / (ticks - 1)
        py = scale(value, y_min, y_max, y + h, y)
        lines.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x+w}" y2="{py:.1f}" class="grid"/>')
        label = f"{10 ** value:.1e}" if log_y else f"{value:.2f}"
        lines.append(f'<text x="{x-8}" y="{py+4:.1f}" text-anchor="end" class="small">{label}</text>')

    for epoch in [1, 5, 10, 15, 20]:
        px = scale(epoch, 1, 20, x, x + w)
        lines.append(f'<text x="{px:.1f}" y="{y+h+22}" text-anchor="middle" class="small">{epoch}</text>')
    lines.append(f'<text x="{x+w/2}" y="{y+h+48}" text-anchor="middle" class="label">epoch</text>')
    lines.append(
        f'<text x="{x-54}" y="{y+h/2}" transform="rotate(-90 {x-54} {y+h/2})" '
        f'text-anchor="middle" class="label">{html.escape(y_label)}</text>'
    )


def render_line(grouped, group_key, metric, filename, y_label, lower=None, upper=None, log_y=False):
    strategies = GROUPS[group_key]
    values = []
    for strategy in strategies:
        for row in grouped[strategy]:
            value = row[metric]
            values.append(value)
    if log_y:
        import math

        transformed_values = [math.log10(value) for value in values]
        y_min, y_max = padded_range(transformed_values)
    else:
        y_min, y_max = padded_range(values, lower=lower, upper=upper)

    width, height = 920, 430
    x, y, w, h = 82, 62, 650, 275
    lines = svg_start(width, height)
    draw_axes(lines, x, y, w, h, f"{TITLES[group_key]} - {y_label}", y_label, y_min, y_max, log_y=log_y)

    for strategy in strategies:
        points = []
        for row in grouped[strategy]:
            value = row[metric]
            if log_y:
                import math

                value = math.log10(value)
            px = scale(row["epoch"], 1, 20, x, x + w)
            py = scale(value, y_min, y_max, y + h, y)
            points.append(f"{px:.1f},{py:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[strategy]}" stroke-width="2.4"/>'
        )
        for row in grouped[strategy][::4]:
            value = row[metric]
            if log_y:
                import math

                value = math.log10(value)
            px = scale(row["epoch"], 1, 20, x, x + w)
            py = scale(value, y_min, y_max, y + h, y)
            lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{COLORS[strategy]}"/>')

    legend_x, legend_y = 760, 86
    for i, strategy in enumerate(strategies):
        ly = legend_y + i * 28
        lines.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x+24}" y2="{ly}" stroke="{COLORS[strategy]}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x+32}" y="{ly+4}" class="small">{html.escape(strategy)}</text>')

    write_svg(filename, lines)


def render_gap(grouped):
    render_line(
        grouped,
        "regularization_bn_dropout",
        "gap",
        "regularization_bn_dropout_train_val_gap.svg",
        "train - validation accuracy gap (%p)",
        lower=0,
    )


def render_bar_final_gap(grouped):
    strategies = GROUPS["regularization_bn_dropout"]
    values = [grouped[strategy][-1]["gap"] for strategy in strategies]
    y_min, y_max = padded_range(values, lower=0)
    width, height = 820, 390
    x, y, w, h = 82, 62, 620, 235
    lines = svg_start(width, height)
    draw_axes(lines, x, y, w, h, "BatchNorm and Dropout - final gap", "gap (%p)", y_min, y_max)
    group_w = w / len(strategies)
    bar_w = 48
    for index, strategy in enumerate(strategies):
        value = grouped[strategy][-1]["gap"]
        cx = x + group_w * index + group_w / 2
        bx = cx - bar_w / 2
        by = scale(value, y_min, y_max, y + h, y)
        lines.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{y+h-by:.1f}" '
            f'fill="{COLORS[strategy]}" rx="4"/>'
        )
        lines.append(f'<text x="{cx:.1f}" y="{by-6:.1f}" text-anchor="middle" class="small">{value:.2f}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{y+h+24}" text-anchor="middle" class="small">{html.escape(strategy)}</text>')
    write_svg("regularization_bn_dropout_final_gap.svg", lines)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    grouped = load_rows()
    for rows in grouped.values():
        for row in rows:
            row["gap"] = row["train_acc"] - row["val_acc"]

    render_line(grouped, "optimizer_sgd_vs_adam", "val_acc", "optimizer_sgd_vs_adam_val_accuracy.svg", "validation accuracy (%)", lower=80, upper=100)
    render_line(grouped, "optimizer_sgd_vs_adam", "val_loss", "optimizer_sgd_vs_adam_val_loss.svg", "validation loss", lower=0)
    render_line(grouped, "regularization_bn_dropout", "val_acc", "regularization_bn_dropout_val_accuracy.svg", "validation accuracy (%)", lower=95, upper=100)
    render_line(grouped, "regularization_bn_dropout", "val_loss", "regularization_bn_dropout_val_loss.svg", "validation loss", lower=0)
    render_gap(grouped)
    render_bar_final_gap(grouped)
    render_line(grouped, "learning_rate_comparison", "val_acc", "learning_rate_comparison_val_accuracy.svg", "validation accuracy (%)", lower=95, upper=100)
    render_line(grouped, "learning_rate_comparison", "val_loss", "learning_rate_comparison_val_loss.svg", "validation loss", lower=0)
    render_line(grouped, "learning_rate_comparison", "lr", "learning_rate_comparison_lr_schedule.svg", "learning rate", log_y=True)


if __name__ == "__main__":
    main()
