"""Build comparison tables and dependency-free SVG plots for CNN-EXP-036–044."""

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "experiments" / "CNN_EXP036_044_COMPARISON"
PLOTS = OUT / "plots"


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#263238}.title{font-size:22px;font-weight:600}.axis{font-size:12px}.label{font-size:11px}.grid{stroke:#dfe4e8;stroke-width:1}.base{stroke:#455a64;stroke-width:1.2}.legend{font-size:12px}</style>',
        f'<text x="40" y="34" class="title">{escape(title)}</text>',
    ]


def write_svg(name: str, lines: list[str]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    lines.append("</svg>")
    (PLOTS / name).write_text("\n".join(lines), encoding="utf-8")


def bar_panels(summary: pd.DataFrame) -> None:
    width, height = 1280, 930
    lines = svg_start(width, height, "CNN experiments 036–044: main validation metrics")
    colors = {"adopt": "#2a9d8f", "iterate": "#e9c46a", "reject": "#e76f51"}
    specs = [
        ("validation_ap_5dp", "Validation AP (5 d.p.)"),
        ("presence_spearman_raw", "Presence EPIC Spearman (raw)"),
        ("intensity_spearman_raw", "Intensity EPIC Spearman (raw)"),
    ]
    left, right, panel_h = 100, 30, 235
    chart_w = width - left - right
    for panel_i, (column, title) in enumerate(specs):
        top = 70 + panel_i * 270
        bottom = top + panel_h
        values = summary[column].astype(float)
        ymax = float(values.max()) * 1.18
        lines.append(f'<text x="{left}" y="{top - 15}" font-size="16" font-weight="600">{escape(title)}</text>')
        for tick in range(5):
            val = ymax * tick / 4
            y = bottom - panel_h * tick / 4
            lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>')
            lines.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="axis">{val:.3f}</text>')
        lines.append(f'<line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" class="base"/>')
        slot = chart_w / len(summary)
        bar_w = slot * 0.62
        for i, row in summary.reset_index(drop=True).iterrows():
            value = float(row[column])
            bar_h = panel_h * value / ymax
            x = left + slot * i + (slot - bar_w) / 2
            y = bottom - bar_h
            color = colors[str(row["decision"])]
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" stroke="#37474f" stroke-width="0.5"/>')
            lines.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" class="label">{value:.3f}</text>')
            if panel_i == 2:
                lines.append(f'<text x="{x+bar_w/2:.1f}" y="{bottom+19}" text-anchor="middle" class="axis">{escape(str(row["experiment"]))}</text>')
    legend_y = 895
    for i, (label, color) in enumerate((('adopt', '#2a9d8f'), ('iterate', '#e9c46a'), ('reject', '#e76f51'))):
        x = 420 + i * 145
        lines.append(f'<rect x="{x}" y="{legend_y-12}" width="16" height="16" fill="{color}"/>')
        lines.append(f'<text x="{x+23}" y="{legend_y+1}" class="legend">{label}</text>')
    lines.append('<text x="640" y="920" text-anchor="middle" class="label">EXP037/038 are frozen-head interventions; other rows are end-to-end backbones.</text>')
    write_svg("01_main_metrics_exp036_044.svg", lines)


def line_panel(lines, frame, datasets, title, ylabel) -> None:
    left, top, panel_w, panel_h = frame
    bottom = top + panel_h
    xs = [x for _, points, _, _ in datasets for x, _ in points]
    ys = [y for _, points, _, _ in datasets for _, y in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max((ymax - ymin) * 0.2, 0.004)
    ymin, ymax = max(0.0, ymin - pad), ymax + pad

    def sx(value):
        return left + (value - xmin) / (xmax - xmin) * panel_w

    def sy(value):
        return bottom - (value - ymin) / (ymax - ymin) * panel_h

    lines.append(f'<text x="{left}" y="{top-18}" font-size="16" font-weight="600">{escape(title)}</text>')
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = sy(value)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+panel_w}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" class="axis">{value:.3f}</text>')
    for tick in (10, 20, 30, 40, 50):
        if xmin <= tick <= xmax:
            x = sx(tick)
            lines.append(f'<text x="{x:.1f}" y="{bottom+20}" text-anchor="middle" class="axis">{tick}k</text>')
    lines.append(f'<text x="{left-55}" y="{top+panel_h/2}" text-anchor="middle" class="axis" transform="rotate(-90 {left-55} {top+panel_h/2})">{escape(ylabel)}</text>')
    for _, points, color, dashed in datasets:
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        for x, y in points:
            lines.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')


def scaling_plot(summary: pd.DataFrame) -> None:
    comparable = summary.query("model_kind == 'backbone'").copy()
    lines = svg_start(1460, 600, "Comparable RF1029 backbones: width, training budget and LR schedule")
    metrics = [
        ("validation_ap_5dp", "Validation AP"),
        ("presence_spearman_raw", "Presence Spearman"),
        ("intensity_spearman_raw", "Intensity Spearman"),
    ]
    for panel_i, (metric, title) in enumerate(metrics):
        datasets = []
        for width_value, color in ((64, "#577590"), (128, "#43aa8b")):
            part = comparable[comparable["channels"] == width_value].sort_values("steps")
            points = [(float(r.steps / 1000), float(r[metric])) for _, r in part.iterrows()]
            if points:
                datasets.append((f"WIDTH{width_value}", points, color, False))
        part = comparable[(comparable["channels"] == 256) & (comparable["experiment"] != "EXP044")].sort_values("steps")
        datasets.append(("WIDTH256 constant", [(float(r.steps / 1000), float(r[metric])) for _, r in part.iterrows()], "#f8961e", False))
        part = comparable[comparable["experiment"].isin(["EXP042", "EXP044"])].sort_values("steps")
        datasets.append(("WIDTH256 cosine", [(float(r.steps / 1000), float(r[metric])) for _, r in part.iterrows()], "#9c6644", True))
        line_panel(lines, (90 + panel_i * 465, 100, 365, 370), datasets, title, "Metric value")
    for i, (label, color) in enumerate((('WIDTH64', '#577590'), ('WIDTH128', '#43aa8b'), ('WIDTH256 constant', '#f8961e'), ('WIDTH256 cosine', '#9c6644'))):
        x = 325 + i * 230
        lines.append(f'<line x1="{x}" y1="545" x2="{x+28}" y2="545" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{x+36}" y="549" class="legend">{escape(label)}</text>')
    write_svg("02_backbone_scaling_and_training.svg", lines)


def cosine_plot(curve: pd.DataFrame) -> None:
    lines = svg_start(1050, 600, "EXP044 cosine continuation: validation metrics by checkpoint")
    datasets = [
        ("AP", [(float(r.step / 1000), float(r.presence_AP_5dp)) for _, r in curve.iterrows()], "#277da1", False),
        ("Presence Spearman", [(float(r.step / 1000), float(r.presence_EPIC_Spearman_raw)) for _, r in curve.iterrows()], "#43aa8b", False),
        ("Intensity Spearman", [(float(r.step / 1000), float(r.intensity_EPIC_Spearman_raw)) for _, r in curve.iterrows()], "#f8961e", False),
    ]
    line_panel(lines, (105, 95, 870, 390), datasets, "AP and Spearman across checkpoints", "Metric value")
    for i, (label, color) in enumerate((('AP', '#277da1'), ('Presence Spearman', '#43aa8b'), ('Intensity Spearman', '#f8961e'))):
        x = 205 + i * 250
        lines.append(f'<line x1="{x}" y1="555" x2="{x+30}" y2="555" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{x+38}" y="559" class="legend">{escape(label)}</text>')
    write_svg("03_exp044_validation_curve.svg", lines)


def main() -> None:
    rows = [
        ("EXP036", "WIDTH64 10k", 64, 10_000, "backbone", 0.033996892, 0.177041367, 0.188708663, "adopt"),
        ("EXP037", "Hard-FP reranker", 64, 10_000, "head", 0.008222070, 0.102930993, 0.188708663, "reject"),
        ("EXP038", "LOW_TP residual", 64, 10_000, "head", 0.034072849, 0.177058339, 0.188708663, "reject"),
        ("EXP039", "WIDTH128 10k", 128, 10_000, "backbone", 0.033810685, 0.173279658, 0.182840765, "iterate"),
        ("EXP040", "WIDTH256 10k", 256, 10_000, "backbone", 0.032517261, 0.170113295, 0.151180565, "iterate"),
        ("EXP041", "WIDTH128 25k", 128, 25_000, "backbone", 0.074412694, 0.188400388, 0.205569088, "adopt"),
        ("EXP042", "WIDTH256 25k", 256, 25_000, "backbone", 0.075671327, 0.187675476, 0.208023280, "adopt"),
        ("EXP043", "WIDTH256 50k constant", 256, 50_000, "backbone", 0.071507846, 0.187260523, 0.206926167, "reject"),
        ("EXP044", "WIDTH256 50k cosine", 256, 50_000, "backbone", 0.093361726, 0.195896953, 0.226627737, "adopt"),
    ]
    columns = ["experiment", "label", "channels", "steps", "model_kind", "validation_ap_5dp", "presence_spearman_raw", "intensity_spearman_raw", "decision"]
    summary = pd.DataFrame(rows, columns=columns)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "cnn_exp036_044_main_metrics.csv", index=False)
    curve_path = ROOT / "artifacts" / "experiments" / "CNN_EXP044_WIDTH256_25K_TO_50K_COSINE_LR" / "run_001" / "cosine_validation_curve_25k_to_50k.csv"
    curve = pd.read_csv(curve_path)
    curve.to_csv(OUT / "exp044_cosine_validation_curve.csv", index=False)
    bar_panels(summary)
    scaling_plot(summary)
    cosine_plot(curve)


if __name__ == "__main__":
    main()
