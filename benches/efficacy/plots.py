#!/usr/bin/env python3
"""Dependency-free SVG plotting for the Reflex efficacy study (REF-176 Phase 3).

matplotlib is not installed in the benchmark environment, and adding it would
make the plots non-reproducible across machines (font/backend drift). SVG is a
better fit here anyway: it is deterministic text, diff-able, renders in browsers
and GitHub/issue attachments, and needs zero dependencies.

Each function returns an SVG document as a string. The helpers are intentionally
minimal — linear axes, rectangles, circles, text — which is all the summary
figures the acceptance criteria call for actually need.
"""
from __future__ import annotations

import html
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Low-level SVG canvas
# ---------------------------------------------------------------------------
@dataclass
class Canvas:
    width: int
    height: int
    _parts: list[str] | None = None

    def __post_init__(self) -> None:
        self._parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            f'font-family="ui-sans-serif, system-ui, sans-serif">',
            f'<rect width="{self.width}" height="{self.height}" fill="#ffffff"/>',
        ]

    def rect(self, x, y, w, h, fill, stroke="none", sw=0, opacity=1.0):
        self._parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'opacity="{opacity}"/>')

    def line(self, x1, y1, x2, y2, stroke="#333", sw=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def circle(self, cx, cy, r, fill, stroke="none", sw=0, opacity=1.0):
        self._parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>')

    def text(self, x, y, s, size=12, fill="#222", anchor="start", weight="normal"):
        self._parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(str(s))}</text>')

    def done(self) -> str:
        return "\n".join(self._parts + ["</svg>"]) + "\n"


# Arm → colour (colour-blind-friendly, stable across all figures).
ARM_COLORS = {
    "A": "#4C72B0",      # control blue
    "B": "#DD8452",      # treatment orange
    "C": "#55A868",      # green
    "Bprime": "#C44E52",  # red
    "Beager": "#8172B2",  # purple
}


def _color(arm: str) -> str:
    return ARM_COLORS.get(arm, "#888888")


# ---------------------------------------------------------------------------
# Figure: per-task token-ratio distribution (H1)
# ---------------------------------------------------------------------------
def ratio_distribution_svg(title: str, pair_label: str,
                           ratios: list[float], point: float,
                           ci_low: float, ci_high: float) -> str:
    """Strip plot of per-task ratios (treatment/control) with the median + CI.

    A dashed line at ratio=1.0 is the parity reference: points/CI left of it
    mean the treatment used *fewer* tokens; right means *more*.
    """
    W, H = 640, 300
    ml, mr, mt, mb = 60, 30, 56, 60
    plot_w = W - ml - mr
    plot_h = H - mt - mb
    c = Canvas(W, H)
    c.text(W / 2, 24, title, size=15, anchor="middle", weight="bold")
    c.text(W / 2, 42, pair_label, size=12, anchor="middle", fill="#666")

    vals = ratios + [point, ci_low, ci_high, 1.0]
    lo = min(v for v in vals if v == v)
    hi = max(v for v in vals if v == v)
    span = hi - lo or 1.0
    lo -= span * 0.12
    hi += span * 0.12

    def sx(v: float) -> float:
        return ml + (v - lo) / (hi - lo) * plot_w

    # Axis
    axis_y = mt + plot_h * 0.62
    c.line(ml, axis_y, ml + plot_w, axis_y, stroke="#999", sw=1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = lo + frac * (hi - lo)
        x = sx(v)
        c.line(x, axis_y - 4, x, axis_y + 4, stroke="#999", sw=1)
        c.text(x, axis_y + 20, f"{v:.2f}", size=10, anchor="middle", fill="#666")

    # Parity reference at 1.0
    if lo <= 1.0 <= hi:
        x1 = sx(1.0)
        c.line(x1, mt + 6, x1, axis_y, stroke="#c00", sw=1, dash="4 3")
        c.text(x1, mt + 2, "parity (1.0)", size=10, anchor="middle", fill="#c00")

    # CI band + median
    cy = axis_y - 34
    c.line(sx(ci_low), cy, sx(ci_high), cy, stroke="#333", sw=2)
    c.line(sx(ci_low), cy - 5, sx(ci_low), cy + 5, stroke="#333", sw=2)
    c.line(sx(ci_high), cy - 5, sx(ci_high), cy + 5, stroke="#333", sw=2)
    c.circle(sx(point), cy, 5, fill="#111")
    c.text(sx(point), cy - 12, f"median {point:.3f}", size=11, anchor="middle",
           weight="bold")
    c.text(sx(ci_low), cy + 20, f"[{ci_low:.3f}, {ci_high:.3f}] 95% CI",
           size=10, anchor="start", fill="#555")

    # Per-task points (jittered deterministically by index)
    for i, r in enumerate(ratios):
        jitter = ((i % 5) - 2) * 4.0
        c.circle(sx(r), axis_y - 2 + jitter, 4, fill="#DD8452", stroke="#8a4",
                 sw=0, opacity=0.75)
    c.text(ml, H - 16, f"n = {len(ratios)} task(s); each dot = one task's "
           f"median ratio", size=10, fill="#666")
    return c.done()


# ---------------------------------------------------------------------------
# Figure: grouped bar chart (per-arm rates: success / precision / recall)
# ---------------------------------------------------------------------------
def grouped_bars_svg(title: str, groups: list[str],
                     series: list[tuple[str, list[float]]],
                     ymax: float = 1.0, ylabel: str = "rate") -> str:
    """Grouped bars. ``series`` = [(label, [value per group]), ...]."""
    W, H = 640, 320
    ml, mr, mt, mb = 56, 20, 56, 70
    plot_w = W - ml - mr
    plot_h = H - mt - mb
    c = Canvas(W, H)
    c.text(W / 2, 26, title, size=15, anchor="middle", weight="bold")

    # Y axis
    c.line(ml, mt, ml, mt + plot_h, stroke="#999", sw=1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = ymax * frac
        y = mt + plot_h - frac * plot_h
        c.line(ml - 4, y, ml, y, stroke="#999", sw=1)
        c.text(ml - 8, y + 4, f"{yv:.2f}", size=10, anchor="end", fill="#666")
    c.text(ml - 40, mt + plot_h / 2, ylabel, size=11, anchor="middle", fill="#666")

    n_groups = len(groups)
    n_series = len(series)
    gw = plot_w / max(1, n_groups)
    bw = gw / (n_series + 1)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for gi, g in enumerate(groups):
        gx = ml + gi * gw
        for si, (slabel, vals) in enumerate(series):
            v = vals[gi]
            if v != v:  # NaN
                continue
            bx = gx + (si + 0.5) * bw + bw * 0.25
            bh = (v / ymax) * plot_h if ymax else 0
            by = mt + plot_h - bh
            c.rect(bx, by, bw * 0.8, bh, fill=palette[si % len(palette)])
            c.text(bx + bw * 0.4, by - 4, f"{v:.2f}", size=9, anchor="middle",
                   fill="#333")
        c.text(gx + gw / 2, mt + plot_h + 18, g, size=11, anchor="middle")

    # Legend
    lx = ml
    ly = H - 24
    for si, (slabel, _) in enumerate(series):
        c.rect(lx, ly - 9, 11, 11, fill=palette[si % len(palette)])
        c.text(lx + 16, ly, slabel, size=10, fill="#444")
        lx += 28 + len(slabel) * 7
    return c.done()


# ---------------------------------------------------------------------------
# Figure: forest plot of the primary endpoint (cold vs warm ratio + CI)
# ---------------------------------------------------------------------------
def forest_svg(title: str, rows: list[tuple[str, float, float, float]]) -> str:
    """Forest plot. ``rows`` = [(label, point, ci_low, ci_high), ...].

    Used for the pre-registered primary endpoint reported cold + warm, with a
    parity line at ratio = 1.0.
    """
    W = 640
    row_h = 46
    mt, mb, ml, mr = 60, 46, 150, 40
    H = mt + mb + row_h * len(rows)
    plot_w = W - ml - mr
    c = Canvas(W, H)
    c.text(W / 2, 26, title, size=15, anchor="middle", weight="bold")
    c.text(W / 2, 44, "ratio = treatment tokens / control tokens (lower = better)",
           size=11, anchor="middle", fill="#666")

    finite = [v for (_, p, lo, hi) in rows for v in (p, lo, hi) if v == v]
    lo = min(finite + [1.0])
    hi = max(finite + [1.0])
    span = hi - lo or 1.0
    lo -= span * 0.15
    hi += span * 0.15

    def sx(v: float) -> float:
        return ml + (v - lo) / (hi - lo) * plot_w

    # Parity line
    if lo <= 1.0 <= hi:
        x1 = sx(1.0)
        c.line(x1, mt - 6, x1, H - mb + 6, stroke="#c00", sw=1, dash="4 3")
        c.text(x1, mt - 10, "parity", size=10, anchor="middle", fill="#c00")

    # X ticks
    for frac in (0.0, 0.5, 1.0):
        v = lo + frac * (hi - lo)
        x = sx(v)
        c.text(x, H - mb + 22, f"{v:.2f}", size=10, anchor="middle", fill="#666")

    for ri, (label, point, ci_low, ci_high) in enumerate(rows):
        y = mt + ri * row_h + row_h / 2
        c.text(ml - 12, y + 4, label, size=12, anchor="end")
        if ci_low == ci_low and ci_high == ci_high:
            c.line(sx(ci_low), y, sx(ci_high), y, stroke="#333", sw=2)
            c.line(sx(ci_low), y - 5, sx(ci_low), y + 5, stroke="#333", sw=2)
            c.line(sx(ci_high), y - 5, sx(ci_high), y + 5, stroke="#333", sw=2)
        if point == point:
            c.circle(sx(point), y, 5, fill="#111")
            c.text(sx(point), y - 10, f"{point:.3f}", size=10, anchor="middle",
                   weight="bold")
    return c.done()
