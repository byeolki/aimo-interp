#!/usr/bin/env python3
"""Build report/report.html from committed experiment outputs (stdlib only).

Render to PDF with any Chromium print-to-PDF (the PDF in this folder was made that way).
Numbers that come from csv files are read here; the rest are quoted from the RESULTS.md files.
"""

import csv
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "report.html"


def layer_curves() -> dict[str, list[tuple[float, float]]]:
    rows = list(csv.DictReader((ROOT / "experiments" / "01-layer-probe" / "cv.csv").open()))
    wanted = {
        "shared probe, DeepSeek-R1-8B, last token": ("shared", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "last"),
        "shared probe, Qwen3.5-4B, mean pool": ("shared", "Qwen/Qwen3.5-4B", "mean"),
        "self probe, Qwen3.5-4B, last token (12 rows)": ("self", "Qwen/Qwen3.5-4B", "last"),
    }
    curves = {}
    for name, (config, encoder, pooling) in wanted.items():
        points = [
            (float(r["rel_depth"]), float(r["bal_acc"]))
            for r in rows
            if r["config"] == config and r["encoder"] == encoder and r["pooling"] == pooling and r["subset"] == "official"
        ]
        curves[name] = sorted(points)
    return curves


def line_chart(curves: dict, baselines: dict[str, float], width=620, height=250) -> str:
    left, right, top, bottom = 48, 190, 12, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.35, 1.0
    colors = ["#1f4e79", "#c0504d", "#7f7f7f"]

    def x(v):
        return left + v * plot_w

    def y(v):
        return top + (y_max - v) / (y_max - y_min) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial" font-size="10">']
    for tick in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 3:.1f}" text-anchor="end">{tick:.1f}</text>')
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{x(tick):.1f}" y="{top + plot_h + 14}" text-anchor="middle">{tick:.2f}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 4}" text-anchor="middle">relative layer depth</text>')
    parts.append(f'<text transform="translate(12,{top + plot_h / 2}) rotate(-90)" text-anchor="middle">balanced accuracy (CV)</text>')
    for index, (label, value) in enumerate(baselines.items()):
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y(value):.1f}" y2="{y(value):.1f}" stroke="#999" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{left + plot_w + 6}" y="{y(value) + 3 + index * 10:.1f}" fill="#666">{html.escape(label)}</text>')
    for (label, points), color in zip(curves.items(), colors):
        path = " ".join(f"{'M' if i == 0 else 'L'}{x(px):.1f},{y(py):.1f}" for i, (px, py) in enumerate(points))
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.6"/>')
    for index, (label, color) in enumerate(zip(curves, colors)):
        ly = top + 8 + index * 40
        parts.append(f'<line x1="{left + plot_w + 6}" x2="{left + plot_w + 22}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="2"/>')
        words = label.split(", ")
        for j, word in enumerate(words):
            parts.append(f'<text x="{left + plot_w + 26}" y="{ly + 4 + j * 11}">{html.escape(word)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(bars: list[tuple[str, float, float, str]], width=620, height=210) -> str:
    left, top, bar_h, gap = 250, 10, 18, 9
    plot_w = width - left - 50

    def x(v):
        return left + v * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica, Arial" font-size="10">']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" y2="{top + len(bars) * (bar_h + gap)}" stroke="#eee"/>')
        parts.append(f'<text x="{x(tick):.1f}" y="{top + len(bars) * (bar_h + gap) + 12}" text-anchor="middle">{tick:.2f}</text>')
    chance_x = x(0.5)
    parts.append(f'<line x1="{chance_x:.1f}" x2="{chance_x:.1f}" y1="{top - 4}" y2="{top + len(bars) * (bar_h + gap)}" stroke="#c00" stroke-dasharray="3 3"/>')
    for index, (label, low, high, color) in enumerate(bars):
        y0 = top + index * (bar_h + gap)
        parts.append(f'<text x="{left - 8}" y="{y0 + bar_h - 5}" text-anchor="end">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y0}" width="{x(low) - left:.1f}" height="{bar_h}" fill="{color}"/>')
        if high > low:
            parts.append(f'<rect x="{x(low):.1f}" y="{y0}" width="{x(high) - x(low):.1f}" height="{bar_h}" fill="{color}" opacity="0.45"/>')
        text = f"{low:.2f}" if high == low else f"{low:.2f}-{high:.2f}"
        parts.append(f'<text x="{x(high) + 5:.1f}" y="{y0 + bar_h - 5}">{text}</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    figure1 = line_chart(layer_curves(), {"per-model majority 0.51": 0.507, "surface text 0.48": 0.483})
    figure2 = bar_chart([
        ("model identity only", 0.68, 0.68, "#7f7f7f"),
        ("self-probe, own threshold", 0.52, 0.59, "#1f4e79"),
        ("self-probe, per-problem rank (raw)", 0.63, 0.69, "#1f4e79"),
        ("self-probe, within-model median", 0.48, 0.49, "#c0504d"),
        ("self-probe, per-problem rank (z-scored)", 0.42, 0.49, "#c0504d"),
        ("counterfactual rule (held-out halves)", 0.64, 0.66, "#548235"),
    ])
    template = (ROOT / "report" / "template.html").read_text(encoding="utf-8")
    OUT.write_text(template.replace("{{FIGURE1}}", figure1).replace("{{FIGURE2}}", figure2), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
