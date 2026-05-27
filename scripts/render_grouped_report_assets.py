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

DISPLAY_NAMES = {
    "adam_baseline": "Adam baseline",
    "sgd_lr_0_01": "SGD lr=0.01",
    "no_batchnorm": "No BatchNorm",
    "no_dropout": "No Dropout",
    "adam_lr_0_01": "Adam lr=0.01",
    "adam_lr_decay": "Adam lr decay",
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
    strategies = GROUPS["regularization_bn_dropout"]
    y_min, y_max = 95, 100
    width, height = 1040, 440
    panel_y, panel_w, panel_h = 92, 245, 230
    panel_gap = 70
    start_x = 78
    lines = svg_start(width, height)
    lines.append('<text x="78" y="34" class="title">Train vs validation accuracy - gap context</text>')
    lines.append(
        '<text x="78" y="56" class="small">'
        'A larger gap is only risky when validation accuracy stalls or validation loss worsens.'
        '</text>'
    )

    legend_x, legend_y = 760, 38
    lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+28}" y2="{legend_y}" stroke="#6b7280" stroke-width="2.2" stroke-dasharray="5 4"/>')
    lines.append(f'<text x="{legend_x+36}" y="{legend_y+4}" class="small">train accuracy</text>')
    lines.append(f'<line x1="{legend_x}" y1="{legend_y+22}" x2="{legend_x+28}" y2="{legend_y+22}" stroke="#111827" stroke-width="2.4"/>')
    lines.append(f'<text x="{legend_x+36}" y="{legend_y+26}" class="small">validation accuracy</text>')

    for index, strategy in enumerate(strategies):
        x = start_x + index * (panel_w + panel_gap)
        color = COLORS[strategy]
        rows = grouped[strategy]
        lines.append(f'<text x="{x}" y="{panel_y-18}" class="label">{html.escape(DISPLAY_NAMES[strategy])}</text>')
        lines.append(f'<rect x="{x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="#ffffff" stroke="#d1d5db"/>')

        for tick in [95, 96, 97, 98, 99, 100]:
            py = scale(tick, y_min, y_max, panel_y + panel_h, panel_y)
            lines.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x+panel_w}" y2="{py:.1f}" class="grid"/>')
            if index == 0:
                lines.append(f'<text x="{x-8}" y="{py+4:.1f}" text-anchor="end" class="small">{tick}</text>')

        for epoch in [1, 10, 20]:
            px = scale(epoch, 1, 20, x, x + panel_w)
            lines.append(f'<text x="{px:.1f}" y="{panel_y+panel_h+20}" text-anchor="middle" class="small">{epoch}</text>')

        train_points = []
        val_points = []
        for row in rows:
            px = scale(row["epoch"], 1, 20, x, x + panel_w)
            train_py = scale(row["train_acc"], y_min, y_max, panel_y + panel_h, panel_y)
            val_py = scale(row["val_acc"], y_min, y_max, panel_y + panel_h, panel_y)
            train_points.append((px, train_py))
            val_points.append((px, val_py))

        gap_polygon = train_points + list(reversed(val_points))
        polygon_points = " ".join(f"{px:.1f},{py:.1f}" for px, py in gap_polygon)
        lines.append(f'<polygon points="{polygon_points}" fill="{color}" opacity="0.12"/>')

        train_line = " ".join(f"{px:.1f},{py:.1f}" for px, py in train_points)
        val_line = " ".join(f"{px:.1f},{py:.1f}" for px, py in val_points)
        lines.append(f'<polyline points="{train_line}" fill="none" stroke="#6b7280" stroke-width="2.2" stroke-dasharray="5 4"/>')
        lines.append(f'<polyline points="{val_line}" fill="none" stroke="{color}" stroke-width="2.6"/>')

        final = rows[-1]
        gap = final["train_acc"] - final["val_acc"]
        lines.append(f'<text x="{x}" y="{panel_y+panel_h+48}" class="small">final val {final["val_acc"]:.2f}%</text>')
        lines.append(f'<text x="{x+128}" y="{panel_y+panel_h+48}" class="small">gap {gap:.2f}%p</text>')

    lines.append(f'<text x="{start_x + panel_w * 1.5 + panel_gap}" y="{height-22}" text-anchor="middle" class="label">epoch</text>')
    lines.append(
        f'<text x="24" y="{panel_y + panel_h/2}" transform="rotate(-90 24 {panel_y + panel_h/2})" '
        f'text-anchor="middle" class="label">accuracy (%)</text>'
    )
    write_svg("regularization_bn_dropout_train_val_gap.svg", lines)


def render_bar_final_gap(grouped):
    strategies = GROUPS["regularization_bn_dropout"]
    values = [grouped[strategy][-1]["gap"] for strategy in strategies]
    y_min, y_max = 0, max(2.0, max(values) * 1.18)
    width, height = 860, 430
    x, y, w, h = 88, 70, 680, 270
    lines = svg_start(width, height)
    lines.append(f'<text x="{x}" y="34" class="title">BatchNorm and Dropout - final gap</text>')
    lines.append(f'<line x1="{x}" y1="{y+h}" x2="{x+w}" y2="{y+h}" class="axis"/>')
    lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" class="axis"/>')

    ticks = [0, 0.5, 1.0, 1.5, 2.0]
    for value in ticks:
        if value > y_max:
            continue
        py = scale(value, y_min, y_max, y + h, y)
        lines.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x+w}" y2="{py:.1f}" class="grid"/>')
        lines.append(f'<text x="{x-10}" y="{py+4:.1f}" text-anchor="end" class="small">{value:.1f}</text>')

    lines.append(
        f'<text x="{x-58}" y="{y+h/2}" transform="rotate(-90 {x-58} {y+h/2})" '
        f'text-anchor="middle" class="label">train - validation accuracy gap (%p)</text>'
    )

    group_w = w / len(strategies)
    bar_w = 72
    for index, strategy in enumerate(strategies):
        value = grouped[strategy][-1]["gap"]
        cx = x + group_w * index + group_w / 2
        bx = cx - bar_w / 2
        by = scale(value, y_min, y_max, y + h, y)
        lines.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{y+h-by:.1f}" '
            f'fill="{COLORS[strategy]}" rx="4"/>'
        )
        lines.append(f'<text x="{cx:.1f}" y="{by-10:.1f}" text-anchor="middle" class="small">{value:.2f}%p</text>')
        lines.append(f'<text x="{cx:.1f}" y="{y+h+28}" text-anchor="middle" class="label">{html.escape(DISPLAY_NAMES[strategy])}</text>')

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
