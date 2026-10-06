"""Turn raw run outputs into the results tables and the consistency chart.

    python -m codecons.analyze [--raw results/raw] [--out results] [--update-readme]

Writes ``results.md`` (tables), ``summary.json`` (numbers and task ids) and
``consistency.png`` (chart). With ``--update-readme`` the tables are also copied
into README.md between the ``RESULTS:START`` / ``RESULTS:END`` markers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from codecons import metrics
from codecons.models import PRECISIONS
from codecons.perturbations import PERTURBATIONS, VARIANTS

REFERENCE = "fp16"
LABELS = {"fp16": "FP16", "int8": "8-bit", "int4": "4-bit"}

# Categorical slots 1-3 of the chart palette, plus a distinct marker per series
# so the series are not told apart by colour alone.
SERIES_STYLE = {
    "fp16": {"color": "#2a78d6", "marker": "o"},
    "int8": {"color": "#eb6834", "marker": "s"},
    "int4": {"color": "#1baf7a", "marker": "D"},
}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", default="results/raw")
    parser.add_argument("--out", default="results")
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()

    outcomes = metrics.load_outcomes(args.raw)
    if not outcomes:
        raise SystemExit(f"no *.jsonl files found in {args.raw}; run `python -m codecons.run` first")
    precisions = [p for p in PRECISIONS if p in outcomes]
    meta = _load_meta(args.raw, precisions)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = build_summary(outcomes, precisions, meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tables = render_tables(summary, precisions)
    (out / "results.md").write_text(tables, encoding="utf-8")
    plot_consistency(summary, precisions, out / "consistency.png")
    print(tables)

    if args.update_readme:
        _update_readme(Path(args.readme), tables)


def build_summary(outcomes: metrics.Outcomes, precisions: List[str], meta: Dict[str, dict]) -> dict:
    summary: dict = {"model": next(iter(meta.values()), {}).get("model"), "precisions": {}}
    for prec in precisions:
        entry: dict = {"memory_footprint_mib": meta.get(prec, {}).get("memory_footprint_mib")}
        entry["pass_at_1"] = {v: _rate(metrics.pass_at_1(outcomes, prec, v)) for v in VARIANTS}
        entry["consistency"] = {}
        for pert in PERTURBATIONS:
            c = metrics.consistency(outcomes, prec, pert)
            entry["consistency"][pert] = {
                **_rate(c.rate),
                "pass_to_fail": c.pass_to_fail,
                "fail_to_pass": c.fail_to_pass,
            }
        entry["consistency"]["overall"] = _rate(metrics.pooled_consistency(outcomes, prec, PERTURBATIONS))

        if prec != REFERENCE and REFERENCE in outcomes:
            entry["vs_fp16"] = {
                "fp16_pass_quant_fail": {
                    v: metrics.disagreement(outcomes, REFERENCE, prec, v) for v in VARIANTS
                },
                "change_induced_failures": {
                    p: metrics.change_induced_failures(outcomes, REFERENCE, prec, p) for p in PERTURBATIONS
                },
                "consistency_test": metrics.compare_consistency(outcomes, REFERENCE, prec, PERTURBATIONS).__dict__,
            }
        summary["precisions"][prec] = entry
    return summary


def render_tables(summary: dict, precisions: List[str]) -> str:
    P = summary["precisions"]
    n_by_variant = {v: P[precisions[0]]["pass_at_1"][v]["n"] for v in VARIANTS}
    lines = []
    if summary.get("model"):
        lines += [f"Model: `{summary['model']}`, greedy decoding, HumanEval ({n_by_variant['original']} problems).", ""]

    lines += ["**pass@1 (%)** per prompt variant", ""]
    header = ["Precision", "Memory (MiB)"] + [f"{v} (n={n_by_variant[v]})" for v in VARIANTS]
    lines += _table_header(header)
    for prec in precisions:
        mem = P[prec]["memory_footprint_mib"]
        cells = [LABELS[prec], f"{mem:.0f}" if mem else "-"]
        cells += [_pct(P[prec]["pass_at_1"][v]["value"]) for v in VARIANTS]
        lines.append(_row(cells))

    lines += ["", "**Consistency (%)**: share of problems whose pass/fail result is unchanged by the prompt change, with 95% Wilson interval", ""]
    lines += _table_header(["Precision"] + list(PERTURBATIONS) + ["overall"])
    for prec in precisions:
        cons = P[prec]["consistency"]
        lines.append(_row([LABELS[prec]] + [_pct_ci(cons[p]) for p in list(PERTURBATIONS) + ["overall"]]))

    lines += ["", "**Outcome flips** after the change (pass→fail / fail→pass)", ""]
    lines += _table_header(["Precision"] + list(PERTURBATIONS))
    for prec in precisions:
        cons = P[prec]["consistency"]
        lines.append(
            _row([LABELS[prec]] + [f"{len(cons[p]['pass_to_fail'])} / {len(cons[p]['fail_to_pass'])}" for p in PERTURBATIONS])
        )

    quantized = [p for p in precisions if "vs_fp16" in P[p]]
    if quantized:
        lines += ["", "**Disagreement with FP16**: problems where FP16 passes but the quantized model fails on the same prompt", ""]
        lines += _table_header(["Precision"] + list(VARIANTS))
        for prec in quantized:
            dis = P[prec]["vs_fp16"]["fp16_pass_quant_fail"]
            lines.append(_row([LABELS[prec]] + [str(len(dis[v])) for v in VARIANTS]))

        lines += [
            "",
            "**Change-induced failures**: FP16 passes before and after the change, the quantized model passes the "
            "original prompt but fails the changed one",
            "",
        ]
        lines += _table_header(["Precision"] + list(PERTURBATIONS) + ["consistency vs FP16 (McNemar p)"])
        for prec in quantized:
            vs = P[prec]["vs_fp16"]
            test = vs["consistency_test"]
            delta = (P[prec]["consistency"]["overall"]["value"] - P[REFERENCE]["consistency"]["overall"]["value"]) * 100
            cells = [LABELS[prec]] + [str(len(vs["change_induced_failures"][p])) for p in PERTURBATIONS]
            cells.append(f"{delta:+.1f} pp (p={test['p_value']:.2f})")
            lines.append(_row(cells))
    return "\n".join(lines) + "\n"


def plot_consistency(summary: dict, precisions: List[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(PERTURBATIONS) + ["overall"]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    width = 0.22
    lows = []
    for j, prec in enumerate(precisions):
        offset = (j - (len(precisions) - 1) / 2) * width
        style = SERIES_STYLE[prec]
        for i, g in enumerate(groups):
            c = summary["precisions"][prec]["consistency"][g]
            lo, hi = c["ci"]
            v = c["value"]
            lows.append(lo)
            x = i + offset
            ax.plot([x, x], [lo * 100, hi * 100], color=style["color"], linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(
                x, v * 100, linestyle="none", marker=style["marker"], markersize=8,
                color=style["color"], markeredgecolor=SURFACE, markeredgewidth=2, zorder=3,
                label=LABELS[prec] if i == 0 else None,
            )

    ax.set_xticks(range(len(groups)))
    n_by_group = [summary["precisions"][precisions[0]]["consistency"][g]["n"] for g in groups]
    ax.set_xticklabels(
        [f"{g.capitalize()}\nn={n}" for g, n in zip(groups, n_by_group)], color=INK, fontsize=10
    )
    ax.axvline(len(groups) - 1.5, color=GRID, linewidth=1)
    floor = max(0, min(lows) * 100 - 3)
    ax.set_ylim(floor, 100.5)
    ax.set_ylabel("Consistency (%)", color=INK_MUTED, fontsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9, length=0)
    ax.tick_params(axis="x", length=0)

    title = "Does the pass/fail result survive a meaning-preserving prompt change?"
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=24)
    ax.text(
        0, 1.02, "Share of HumanEval problems with an unchanged outcome; lines show 95% Wilson intervals",
        transform=ax.transAxes, color=INK_MUTED, fontsize=9,
    )
    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False,
        ncol=len(precisions), fontsize=9, handletextpad=0.3,
    )
    for text in legend.get_texts():
        text.set_color(INK)

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _rate(rate: metrics.Rate) -> dict:
    return {"k": rate.k, "n": rate.n, "value": rate.value, "ci": list(rate.ci)}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}"


def _pct_ci(entry: dict) -> str:
    lo, hi = entry["ci"]
    return f"{entry['value'] * 100:.1f} [{lo * 100:.0f}–{hi * 100:.0f}]"


def _table_header(cells: List[str]) -> List[str]:
    return [_row(cells), _row(["---"] * len(cells))]


def _row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _load_meta(raw_dir: str, precisions: List[str]) -> Dict[str, dict]:
    meta = {}
    for prec in precisions:
        path = Path(raw_dir) / f"{prec}.meta.json"
        if path.exists():
            meta[prec] = json.loads(path.read_text(encoding="utf-8"))
    return meta


def _update_readme(readme: Path, tables: str) -> None:
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"{readme} has no {README_START} ... {README_END} block")
    block = f"{README_START}\n![Consistency by precision and prompt change](results/consistency.png)\n\n{tables}{README_END}"
    readme.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    print(f"updated {readme}")


if __name__ == "__main__":
    main()
