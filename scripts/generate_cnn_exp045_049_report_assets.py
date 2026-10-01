"""Generate dependency-light SVG/CSV assets for the EXP045--EXP049 review."""

from __future__ import annotations

import csv
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "cnn" / "exp045_049_review_2026-10-01"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "reference": "#64748b", "good": "#16a34a", "bad": "#dc2626",
    "joint": "#2563eb", "control": "#f59e0b", "oracle": "#7c3aed",
    "grid": "#dbe3ed", "text": "#172033",
}


def write_csv(name: str, columns: list[str], rows: list[tuple[object, ...]]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def svg_document(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:Arial,"DejaVu Sans",sans-serif;fill:#172033}'
        '.title{font-size:20px;font-weight:700}.axis{font-size:12px}.small{font-size:11px}'
        '.value{font-size:11px;font-weight:700}</style>\n'
        + "\n".join(body) + "\n</svg>\n"
    )


def save_svg(name: str, width: int, height: int, body: list[str]) -> None:
    (OUT / name).write_text(svg_document(width, height, body), encoding="utf-8")


def y_grid(body: list[str], x0: float, x1: float, y0: float, y1: float, maximum: float, ticks: list[float]) -> None:
    for tick in ticks:
        y = y1 - (tick / maximum) * (y1 - y0)
        body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(f'<text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')


overall = [
    ("EXP045", "WIDTH512 50k", 0.097630094, 0.200124323, "direct"),
    ("EXP046B", "frozen R32 correction", 0.098867560, "", "direct"),
    ("EXP047-A", "matched continuation", 0.093970943, 0.192573186, "direct"),
    ("EXP047-B", "risk loss", 0.104211926, 0.202291612, "direct"),
    ("EXP048", "ultra-tail loss", 0.082120732, 0.197690110, "direct"),
    ("049 ctrl 20k", "B, zero condition", 0.046371994, 0.120001431, "joint-pilot"),
    ("049 joint 20k", "A→FiLM→B", 0.092451843, 0.128554933, "joint-pilot"),
    ("049 ctrl 50k", "B, zero condition", 0.073778159, 0.133377696, "joint-pilot"),
    ("049 joint 50k", "A→FiLM→B", 0.104208798, 0.130878357, "joint-pilot"),
]
write_csv("overall_metrics.csv", ["experiment", "variant", "ap5", "epic_spearman", "family"], overall)

body: list[str] = ['<text x="550" y="32" text-anchor="middle" class="title">Последние CNN-эксперименты: genome-wide AP</text>']
x0, x1, y0, y1, max_y = 75, 1070, 65, 420, 0.165
y_grid(body, x0, x1, y0, y1, max_y, [0.00, 0.05, 0.10, 0.15])
target_y = y1 - 0.15 / max_y * (y1 - y0)
body.append(f'<line x1="{x0}" y1="{target_y:.1f}" x2="{x1}" y2="{target_y:.1f}" stroke="#111827" stroke-dasharray="7 5"/>')
body.append(f'<text x="{x1 - 5}" y="{target_y - 7:.1f}" text-anchor="end" class="small">цель AP=0.15</text>')
step = (x1 - x0) / len(overall)
for index, row in enumerate(overall):
    label, _, value, _, _ = row
    color = COLORS["good"] if label == "EXP047-B" else COLORS["bad"] if label == "EXP048" else COLORS["joint"] if "joint" in label else COLORS["control"] if "ctrl" in label else COLORS["reference"]
    bar_x, bar_w = x0 + index * step + 14, step - 28
    bar_y = y1 - value / max_y * (y1 - y0)
    body.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_w:.1f}" height="{y1 - bar_y:.1f}" rx="3" fill="{color}"/>')
    body.append(f'<text x="{bar_x + bar_w / 2:.1f}" y="{bar_y - 7:.1f}" text-anchor="middle" class="value">{value:.4f}</text>')
    cx = bar_x + bar_w / 2
    body.append(f'<text x="{cx:.1f}" y="{y1 + 18}" text-anchor="end" class="small" transform="rotate(-35 {cx:.1f} {y1 + 18})">{escape(label)}</text>')
save_svg("01_overall_ap_comparison.svg", 1100, 525, body)

learning = [
    (20_000, "control", 0.046371994, 0.120001431), (50_000, "control", 0.073778159, 0.133377696),
    (20_000, "joint", 0.092451843, 0.128554933), (50_000, "joint", 0.104208798, 0.130878357),
]
write_csv("exp049_learning_curve.csv", ["steps", "arm", "ap5", "epic_spearman"], learning)
body = ['<text x="550" y="30" text-anchor="middle" class="title">EXP049B: продолжение 20k → 50k</text>']
for px0, px1, title, maximum, metric_index in [(55, 520, "AP", 0.12, 2), (590, 1055, "EPIC Spearman", 0.15, 3)]:
    py0, py1 = 65, 380
    y_grid(body, px0, px1, py0, py1, maximum, [0.0, 0.05, 0.10] if maximum == 0.12 else [0.0, 0.05, 0.10, 0.15])
    body.append(f'<text x="{(px0 + px1) / 2}" y="55" text-anchor="middle" class="axis">{title}</text>')
    for arm, color in [("control", COLORS["control"]), ("joint", COLORS["joint"])]:
        rows = [row for row in learning if row[1] == arm]
        points = []
        for step_value, _, ap, spearman in rows:
            value = ap if metric_index == 2 else spearman
            x = px0 + (step_value - 20_000) / 30_000 * (px1 - px0)
            y = py1 - value / maximum * (py1 - py0)
            points.append((x, y, value))
        body.append(f'<line x1="{points[0][0]}" y1="{points[0][1]:.1f}" x2="{points[1][0]}" y2="{points[1][1]:.1f}" stroke="{color}" stroke-width="3"/>')
        for x, y, value in points:
            body.append(f'<circle cx="{x}" cy="{y:.1f}" r="6" fill="{color}"/>')
            body.append(f'<text x="{x}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value:.4f}</text>')
    body.append(f'<text x="{px0}" y="{py1 + 22}" text-anchor="middle" class="axis">20k</text>')
    body.append(f'<text x="{px1}" y="{py1 + 22}" text-anchor="middle" class="axis">50k</text>')
body.extend([
    f'<rect x="420" y="420" width="14" height="14" fill="{COLORS["control"]}"/><text x="440" y="432" class="axis">control</text>',
    f'<rect x="550" y="420" width="14" height="14" fill="{COLORS["joint"]}"/><text x="570" y="432" class="axis">joint</text>',
])
save_svg("02_exp049_learning_curve.svg", 1100, 465, body)

per_contig = [
    ("NC_064034.1", "control 20k", 0.046607887), ("NC_064041.1", "control 20k", 0.041865308), ("NC_064042.1", "control 20k", 0.050194835),
    ("NC_064034.1", "joint 20k", 0.096129120), ("NC_064041.1", "joint 20k", 0.081820535), ("NC_064042.1", "joint 20k", 0.096262336),
    ("NC_064034.1", "control 50k", 0.072404916), ("NC_064041.1", "control 50k", 0.067657010), ("NC_064042.1", "control 50k", 0.080670010),
    ("NC_064034.1", "joint 50k", 0.108164010), ("NC_064041.1", "joint 50k", 0.094356448), ("NC_064042.1", "joint 50k", 0.107035054),
    ("NC_064034.1", "EXP047-B", 0.103602500), ("NC_064041.1", "EXP047-B", 0.096008027), ("NC_064042.1", "EXP047-B", 0.111698221),
]
write_csv("exp049_per_contig_ap.csv", ["contig", "model", "ap5"], per_contig)
models = ["control 20k", "joint 20k", "control 50k", "joint 50k", "EXP047-B"]
palette = ["#fcd34d", "#93c5fd", COLORS["control"], COLORS["joint"], COLORS["reference"]]
contigs = ["NC_064034.1", "NC_064041.1", "NC_064042.1"]
lookup = {(contig, model): value for contig, model, value in per_contig}
body = ['<text x="550" y="30" text-anchor="middle" class="title">EXP049B: AP по validation contig</text>']
x0, x1, y0, y1, maximum = 70, 1070, 65, 380, 0.125
y_grid(body, x0, x1, y0, y1, maximum, [0.00, 0.04, 0.08, 0.12])
group_w, bar_w = (x1 - x0) / len(contigs), 48
for ci, contig in enumerate(contigs):
    group_start = x0 + ci * group_w + 35
    for mi, (model, color) in enumerate(zip(models, palette)):
        value = lookup[(contig, model)]
        bx, by = group_start + mi * (bar_w + 8), y1 - value / maximum * (y1 - y0)
        body.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{y1 - by:.1f}" fill="{color}"/>')
        body.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 5:.1f}" text-anchor="middle" class="small">{value:.3f}</text>')
    body.append(f'<text x="{group_start + 2.5 * (bar_w + 8):.1f}" y="{y1 + 22}" text-anchor="middle" class="axis">{contig}</text>')
for mi, (model, color) in enumerate(zip(models, palette)):
    lx = 90 + mi * 195
    body.append(f'<rect x="{lx}" y="440" width="14" height="14" fill="{color}"/><text x="{lx + 20}" y="452" class="small">{model}</text>')
save_svg("03_exp049_per_contig_ap.svg", 1100, 480, body)

regional = [
    ("A-v0 20k", 20_000, 0.133937963, 0.134444240, 0.195664970),
    ("A-v1 5k", 5_000, 0.055724698, 0.179522860, 0.108324901),
    ("A-v1 10k", 10_000, 0.095152726, 0.234085601, 0.148124003),
    ("A-v1 20k", 20_000, 0.131841037, 0.252141879, 0.156131558),
]
write_csv("exp049a_regional_metrics.csv", ["checkpoint", "steps", "regional_ap", "active_count_spearman", "empty_suppression_at_99p5_recall"], regional)
body = ['<text x="450" y="30" text-anchor="middle" class="title">Network A: detection ↔ count ordering</text>']
x0, x1, y0, y1 = 80, 860, 65, 430
for tick in [0.05, 0.08, 0.11, 0.14]:
    x = x0 + (tick - 0.04) / 0.11 * (x1 - x0)
    body.append(f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{COLORS["grid"]}"/><text x="{x:.1f}" y="{y1 + 20}" text-anchor="middle" class="axis">{tick:.2f}</text>')
for tick in [0.12, 0.16, 0.20, 0.24, 0.28]:
    y = y1 - (tick - 0.10) / 0.19 * (y1 - y0)
    body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/><text x="{x0 - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.2f}</text>')
for index, (label, _, ap, spearman, _) in enumerate(regional):
    x = x0 + (ap - 0.04) / 0.11 * (x1 - x0)
    y = y1 - (spearman - 0.10) / 0.19 * (y1 - y0)
    color = COLORS["reference"] if index == 0 else COLORS["joint"]
    body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/><text x="{x + 12:.1f}" y="{y - 8:.1f}" class="value">{label}</text>')
body.append(f'<text x="{(x0 + x1) / 2}" y="475" text-anchor="middle" class="axis">Regional AP (R16, held-out fold 0)</text>')
body.append('<text x="18" y="250" text-anchor="middle" class="axis" transform="rotate(-90 18 250)">Spearman count среди active R16</text>')
save_svg("04_exp049a_regional_tradeoff.svg", 900, 500, body)

oracle = [
    (1, 1.000000000, 1.00000), (2, 0.699111615, 0.97325), (4, 0.497603879, 0.93215), (8, 0.369152022, 0.88220),
    (16, 0.289735318, 0.83455), (32, 0.238318258, 0.79210), (64, 0.204895978, 0.75430), (128, 0.181954856, 0.71645),
]
write_csv("oracle_resolution_audit.csv", ["region_width_bp", "oracle_ap5", "fixed_hard_fp_empty_fraction"], oracle)
body = ['<text x="500" y="30" text-anchor="middle" class="title">Label-oracle: AP растёт при механическом раскрытии target</text>']
x0, x1, y0, y1, maximum = 70, 965, 60, 410, 1.05
y_grid(body, x0, x1, y0, y1, maximum, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
points = []
for index, (radius, value, _) in enumerate(oracle):
    x = x0 + index / (len(oracle) - 1) * (x1 - x0)
    y = y1 - value / maximum * (y1 - y0)
    points.append((x, y, radius, value))
body.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y, _, _ in points) + f'" fill="none" stroke="{COLORS["oracle"]}" stroke-width="3"/>')
for x, y, radius, value in points:
    body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{COLORS["oracle"]}"/><text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value:.3f}</text><text x="{x:.1f}" y="{y1 + 22}" text-anchor="middle" class="axis">R{radius}</text>')
baseline_y = y1 - 0.097630094 / maximum * (y1 - y0)
body.append(f'<line x1="{x0}" y1="{baseline_y:.1f}" x2="{x1}" y2="{baseline_y:.1f}" stroke="{COLORS["reference"]}" stroke-dasharray="7 5"/><text x="{x1}" y="{baseline_y - 7:.1f}" text-anchor="end" class="small">EXP045 AP=0.0976</text>')
save_svg("05_oracle_resolution_curve.svg", 1000, 455, body)

print(f"Saved report assets to {OUT}")
