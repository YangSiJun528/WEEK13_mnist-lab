# -*- coding: utf-8 -*-
"""Render SVG charts for the MNIST strategy comparison report."""

from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "experiment_logs" / "strategy_run_2026-05-27.txt"
OUT_DIR = ROOT / "report_assets"

COLORS = {
    "baseline": "#2563eb",
    "high_lr": "#dc2626",
    "lr_decay": "#16a34a",
    "no_dropout": "#9333ea",
    "no_batchnorm": "#ea580c",
    "xavier_init": "#0891b2",
}


def parse_log(path: Path):
    start_re = re.compile(r"^\[(.+)\] start")
    epoch_re = re.compile(
        r"epoch\s+(\d+)/\d+\s+"
        r"lr=([0-9.eE+-]+)\s+"
        r"train_loss=([0-9.eE+-]+)\s+"
        r"train_acc=([0-9.eE+-]+)%\s+"
        r"val_loss=([0-9.eE+-]+)\s+"
        r"val_acc=([0-9.eE+-]+)%"
    )

    results = []
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        start = start_re.search(line)
        if start:
            current = {"name": start.group(1), "history": []}
            results.append(current)
            continue

        match = epoch_re.search(line)
        if match and current is not None:
            current["history"].append(
                {
                    "epoch": int(match.group(1)),
                    "lr": float(match.group(2)),
                    "train_loss": float(match.group(3)),
                    "train_acc": float(match.group(4)),
                    "val_loss": float(match.group(5)),
                    "val_acc": float(match.group(6)),
                }
            )
    return results


def scale(value, source_min, source_max, target_min, target_max):
    if source_max == source_min:
        return (target_min + target_max) / 2
    ratio = (value - source_min) / (source_max - source_min)
    return target_min + ratio * (target_max - target_min)


def svg_header(width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>'
        'text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}'
        '.axis{stroke:#374151;stroke-width:1.2}'
        '.grid{stroke:#e5e7eb;stroke-width:1}'
        '.label{font-size:12px;fill:#4b5563}'
        '.title{font-size:18px;font-weight:700}'
        '.small{font-size:11px;fill:#6b7280}'
        '</style>',
    ]


def write_svg(path: Path, lines):
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")


def draw_axes(lines, x, y, width, height, y_min, y_max, title, y_label):
    lines.append(f'<text x="{x}" y="30" class="title">{html.escape(title)}</text>')
    lines.append(f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" class="axis"/>')
    lines.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" class="axis"/>')
    for i in range(5):
        value = y_min + (y_max - y_min) * i / 4
        py = scale(value, y_min, y_max, y + height, y)
        lines.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x + width}" y2="{py:.1f}" class="grid"/>')
        lines.append(f'<text x="{x - 8}" y="{py + 4:.1f}" text-anchor="end" class="small">{value:.2f}</text>')
    lines.append(
        f'<text x="{x - 52}" y="{y + height / 2}" transform="rotate(-90 {x - 52} {y + height / 2})" '
        f'text-anchor="middle" class="label">{html.escape(y_label)}</text>'
    )


def render_grouped_accuracy(results):
    width, height = 920, 440
    margin_left, margin_right = 80, 30
    plot_x, plot_y = margin_left, 70
    plot_w, plot_h = width - margin_left - margin_right, 260
    y_min, y_max = 98.0, 98.6
    lines = svg_header(width, height)
    draw_axes(lines, plot_x, plot_y, plot_w, plot_h, y_min, y_max, "Validation accuracy comparison", "accuracy (%)")

    group_w = plot_w / len(results)
    bar_w = 24
    for i, result in enumerate(results):
        name = result["name"]
        final = result["history"][-1]["val_acc"]
        best = max(row["val_acc"] for row in result["history"])
        cx = plot_x + group_w * i + group_w / 2
        for j, (label, value, color) in enumerate(
            [("final", final, "#94a3b8"), ("best", best, COLORS[name])]
        ):
            bar_x = cx - bar_w - 4 + j * (bar_w + 8)
            bar_y = scale(value, y_min, y_max, plot_y + plot_h, plot_y)
            lines.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w}" '
                f'height="{plot_y + plot_h - bar_y:.1f}" fill="{color}" rx="3"/>'
            )
            lines.append(
                f'<text x="{bar_x + bar_w / 2:.1f}" y="{bar_y - 6:.1f}" text-anchor="middle" class="small">{value:.2f}</text>'
            )
        lines.append(
            f'<text x="{cx:.1f}" y="{plot_y + plot_h + 28}" text-anchor="middle" class="small">{html.escape(name)}</text>'
        )
    lines.append('<rect x="710" y="34" width="12" height="12" fill="#94a3b8" rx="2"/>')
    lines.append('<text x="728" y="44" class="small">final</text>')
    lines.append('<rect x="780" y="34" width="12" height="12" fill="#2563eb" rx="2"/>')
    lines.append('<text x="798" y="44" class="small">best</text>')
    write_svg(OUT_DIR / "strategy_validation_accuracy.svg", lines)


def render_bar_chart(results, filename, title, y_label, value_fn, y_min, y_max, value_fmt):
    width, height = 920, 420
    plot_x, plot_y = 80, 70
    plot_w, plot_h = 810, 250
    lines = svg_header(width, height)
    draw_axes(lines, plot_x, plot_y, plot_w, plot_h, y_min, y_max, title, y_label)

    group_w = plot_w / len(results)
    bar_w = 48
    for i, result in enumerate(results):
        name = result["name"]
        value = value_fn(result)
        cx = plot_x + group_w * i + group_w / 2
        bar_x = cx - bar_w / 2
        bar_y = scale(value, y_min, y_max, plot_y + plot_h, plot_y)
        lines.append(
            f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w}" '
            f'height="{plot_y + plot_h - bar_y:.1f}" fill="{COLORS[name]}" rx="4"/>'
        )
        lines.append(f'<text x="{cx:.1f}" y="{bar_y - 6:.1f}" text-anchor="middle" class="small">{value_fmt(value)}</text>')
        lines.append(f'<text x="{cx:.1f}" y="{plot_y + plot_h + 28}" text-anchor="middle" class="small">{html.escape(name)}</text>')
    write_svg(OUT_DIR / filename, lines)


def render_line_chart(results, filename, title, y_label, metric, y_min, y_max):
    width, height = 980, 520
    plot_x, plot_y = 80, 70
    plot_w, plot_h = 720, 340
    lines = svg_header(width, height)
    draw_axes(lines, plot_x, plot_y, plot_w, plot_h, y_min, y_max, title, y_label)

    x_min, x_max = 1, 20
    for i in range(1, 21, 3):
        px = scale(i, x_min, x_max, plot_x, plot_x + plot_w)
        lines.append(f'<text x="{px:.1f}" y="{plot_y + plot_h + 24}" text-anchor="middle" class="small">{i}</text>')
    lines.append(f'<text x="{plot_x + plot_w / 2}" y="{plot_y + plot_h + 48}" text-anchor="middle" class="label">epoch</text>')

    for result in results:
        name = result["name"]
        points = []
        for row in result["history"]:
            px = scale(row["epoch"], x_min, x_max, plot_x, plot_x + plot_w)
            py = scale(row[metric], y_min, y_max, plot_y + plot_h, plot_y)
            points.append(f"{px:.1f},{py:.1f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[name]}" stroke-width="2.2"/>'
        )
    legend_x, legend_y = 825, 85
    for i, result in enumerate(results):
        name = result["name"]
        y = legend_y + i * 28
        lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{COLORS[name]}" stroke-width="3"/>')
        lines.append(f'<text x="{legend_x + 32}" y="{y + 4}" class="small">{html.escape(name)}</text>')
    write_svg(OUT_DIR / filename, lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = parse_log(LOG_PATH)
    render_grouped_accuracy(results)
    render_bar_chart(
        results,
        "strategy_final_validation_loss.svg",
        "Final validation loss",
        "loss",
        lambda result: result["history"][-1]["val_loss"],
        0.0,
        0.085,
        lambda value: f"{value:.4f}",
    )
    render_bar_chart(
        results,
        "strategy_train_val_gap.svg",
        "Final train-validation accuracy gap",
        "gap (%p)",
        lambda result: result["history"][-1]["train_acc"] - result["history"][-1]["val_acc"],
        0.0,
        2.0,
        lambda value: f"{value:.2f}",
    )
    render_line_chart(
        results,
        "strategy_validation_accuracy_curves.svg",
        "Validation accuracy over epochs",
        "accuracy (%)",
        "val_acc",
        95.0,
        98.7,
    )
    render_line_chart(
        results,
        "strategy_validation_loss_curves.svg",
        "Validation loss over epochs",
        "loss",
        "val_loss",
        0.045,
        0.165,
    )


if __name__ == "__main__":
    main()
