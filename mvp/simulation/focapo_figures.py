# -*- coding: utf-8 -*-
"""FOCAPO conference figures (distinct from the journal).

fig_h1 — integration superiority (two panels): (a) ablation ladder of pooled
ARI, (b) per-scenario integration gain ΔARI with 95% CI error bars and
Cohen's d. → results/focapo/h1.png

fig_evidence — performance gain + context value (two panels): (a) % ARI
improvement vs each baseline (symlog), (b) context decision quality (ΔARI per
intervention). Recovered from the pre-merge fig9 and slimmed; the Cohen's d
heatmap is dropped because that effect-size content lives in journal Fig 7.
→ results/focapo/evidence.png

The conference paper also shows the Cross-Scenario Performance Comparison figure
(journal fig6_cross → results/cross_scenario.png, journal-unused) and an
8-mode ablation ARI table built from benchmark_summary.json. Renders from the
saved-seed artefacts; run from mvp/simulation (HPC-canonical)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import generate_figures as G

R = G.RESULTS_DIR
SLAB = G.SCENARIO_LABELS
SCEN = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
SCEN5 = SCEN + ["baseline"]
SUMM = json.loads((R / "benchmark_summary.json").read_text()); SUMM = SUMM.get("summary", SUMM)
SIG = json.loads((R / "benchmark_significance.json").read_text())["significance"]
FONT_BUMP = 6   # side-by-side panels at full-page width (enlarged per request)
C_CTX, C_MCP, C_PIRAG = G.COLORS["agribrain"], G.COLORS["mcp_only"], G.COLORS["pirag_only"]

# Enlarged conference typography: titles and axis titles are emphasised on top
# of the global _font_bump. Applied via _bump() and torn down by
# G._font_restore (which restores the pre-bump rcParams snapshot).
SUP_TITLE = 33      # fig.suptitle (point size on the 4-panel figure)
PANEL_TITLE = 33    # (a)/(b) panel titles  (axes.titlesize)
AXIS_TITLE = 30     # x/y axis labels       (axes.labelsize)
# The 4-panel figure is 15 in wide and titled at SUP_TITLE pt. The 2-panel
# figures are wider, so a fixed point size renders smaller once every figure
# is scaled to a common column width. Scale the 2-panel suptitle by width to
# keep the apparent on-page title size identical across all figures.
REF_FIG_W = 15.0    # four-panel (fig_cross) width, in inches


def _bump(delta=FONT_BUMP, panel=PANEL_TITLE, axis=AXIS_TITLE):
    """Global font bump + emphasised title/axis-title overrides."""
    saved = G._font_bump(delta)
    plt.rcParams.update({
        "axes.titlesize": panel,
        "axes.labelsize": axis,
        "figure.titlesize": SUP_TITLE,
    })
    return saved


def _suptitle_flush(fig, axes, text, gap=0.050):
    """Anchor the suptitle flush above the tallest panel title (no slack).

    tight_layout/suptitle interactions leave an unpredictable vertical gap,
    so measure the panel titles' top in figure coordinates after a draw and
    place the bottom-aligned suptitle just above it (it overflows the canvas
    upward; bbox_inches='tight' expands to include it on save)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ytop = max(ax.title.get_window_extent(r).ymax for ax in axes) / fig.bbox.height
    size = SUP_TITLE * fig.get_size_inches()[0] / REF_FIG_W
    fig.suptitle(text, y=ytop + gap, fontsize=size, fontweight="bold", va="bottom")


def _lbl(m):
    """Mode label for the figures, with the AGRI-BRAIN brand spelled canonically."""
    return "AGRI-BRAIN" if m == "agribrain" else G.MODE_LABELS[m]


def _save(fig, name):
    name = name.replace("focapo_", "")
    out = G.RESULTS_DIR / "focapo"; out.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out / (name + ".png")), dpi=G.DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  saved focapo/" + name)


# ===================================================== Fig: integration superiority (2 panels)
def fig_h1():
    saved = _bump()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 7.2))
    # (a) ablation ladder: pooled ARI, ascending — the context step dominates
    LAD = ["static", "no_slca", "hybrid_rl", "no_pinn", "no_context", "pirag_only", "mcp_only", "agribrain"]
    mp, vv, ee = [], [], []
    for m in LAD:
        v = [SUMM[s][m]["ari"]["mean"] for s in SCEN if m in SUMM.get(s, {})]
        if v:
            mp.append(m); vv.append(float(np.mean(v)))
            ee.append(float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    order = np.argsort(vv); mp = [mp[i] for i in order]; vv = [vv[i] for i in order]; ee = [ee[i] for i in order]
    def _lc(m):
        if m == "agribrain": return C_CTX
        if m == "mcp_only": return C_MCP
        if m == "pirag_only": return C_PIRAG
        if m == "no_context": return "#546E7A"
        return "#B0BEC5"
    y = np.arange(len(mp))
    axA.barh(y, vv, xerr=ee, color=[_lc(m) for m in mp], edgecolor="white", height=0.72,
             capsize=G._ERR_CAPSIZE, error_kw=G._ERR_KW)
    axA.set_yticks(y); axA.set_yticklabels([_lbl(m) for m in mp])
    axA.set_xlabel("Mean Adaptive Resilience Index"); axA.set_title("(a) Ablation Ladder")
    _lo = min(v - e for v, e in zip(vv, ee)); _hi = max(v + e for v, e in zip(vv, ee)); _pd = (_hi - _lo) * 0.05
    axA.set_xlim(_lo - _pd, _hi + _pd); G._apply_style(axA)
    # (b) integration gain by scenario (Integrated - No-Context): 95% CI error bars + Cohen's d
    sc = [s for s in SCEN5 if s in SIG]
    dari, elo, ehi, dvals = [], [], [], []
    for s in sc:
        c = SIG[s].get("agribrain_vs_no_context", {}).get("ari", {})
        md = c.get("mean_diff", 0.0); dari.append(md)
        elo.append(max(0.0, md - c.get("mean_diff_ci_low", md)))
        ehi.append(max(0.0, c.get("mean_diff_ci_high", md) - md))
        dvals.append(c.get("cohens_d_pooled", c.get("cohens_d", 0.0)))
    xb = np.arange(len(sc))
    axB.bar(xb, dari, 0.6, color=C_CTX, edgecolor="white", linewidth=0.8,
            yerr=[elo, ehi], capsize=G._ERR_CAPSIZE, error_kw=G._ERR_KW)
    axB.set_xticks(xb); axB.set_xticklabels([SLAB[s] for s in sc], rotation=20, ha="right")
    axB.set_ylabel("ΔARI (Integrated − No-Context)"); axB.set_title("(b) Integration Gain by Scenario")
    top = max(h + e for h, e in zip(dari, ehi)); axB.set_ylim(0, top * 1.24); G._apply_style(axB)
    for xi, h, e, dd in zip(xb, dari, ehi, dvals):
        axB.text(xi, h + e + top * 0.02, f"d={dd:.1f}", ha="center", va="bottom",
                 fontsize=G.ANNOT_FONT_SIZE - 1, fontweight="bold", color="#1F1F1F")
    fig.tight_layout(w_pad=2.4)
    _suptitle_flush(fig, (axA, axB), "Integration Improves Resilience Across Scenarios")
    _save(fig, "focapo_h1"); G._font_restore(saved)


# ===================================================== Fig: performance gain + context value (2 panels)
def fig_evidence():
    saved = _bump()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(16, 7.2))
    BASE = [("static", "vs Static"), ("hybrid_rl", "vs Hybrid RL"), ("no_context", "vs No Context"),
            ("pirag_only", "vs piRAG only"), ("mcp_only", "vs MCP only")]
    scen = [s for s in SCEN5 if s in SIG]

    # (a) % ARI improvement vs each baseline (mean over scenarios, range whiskers).
    # Distinct per-baseline palette (Hybrid RL -> purple) so no two bars collide.
    B_COL = {"static": "#4A4A4A", "hybrid_rl": "#8E24AA", "no_context": "#2E7D32",
             "pirag_only": "#1565C0", "mcp_only": "#F57C00"}
    impr, lo, hi, cols = [], [], [], []
    for b, _ in BASE:
        vals = []
        for s in scen:
            c = SIG[s].get(f"agribrain_vs_{b}", {}).get("ari", {}); md = c.get("mean_diff")
            bm = SUMM.get(s, {}).get(b, {}).get("ari", {}).get("mean")
            if md is not None and bm:
                vals.append(100.0 * md / bm)
        m = float(np.mean(vals)); impr.append(m)
        lo.append(max(0.0, m - min(vals))); hi.append(max(vals) - m); cols.append(B_COL.get(b, "#7E57C2"))
    yb = np.arange(len(BASE))[::-1]   # vs Static on top
    axA.barh(yb, impr, height=0.62, xerr=[lo, hi], color=cols, edgecolor="white", linewidth=0.8,
             capsize=5, error_kw={"lw": 1.5, "ecolor": "#1F1F1F", "alpha": 0.85})
    axA.set_yticks(yb); axA.set_yticklabels([l for _, l in BASE])
    axA.set_xscale("symlog", linthresh=5); axA.set_xlim(0, max(h + e for h, e in zip(impr, hi)) * 1.8)
    axA.set_xlabel("ARI improvement (%)"); axA.set_title("(a) % ARI Improvement vs Baselines")
    G._apply_style(axA)
    for yi, m, e in zip(yb, impr, hi):
        axA.text(m + e, yi, f"  +{m:.1f}%", va="center", ha="left",
                 fontsize=G.ANNOT_FONT_SIZE - 1, fontweight="bold", color="#1F1F1F")

    # (b) context decision quality: ΔARI per context intervention
    MODES = [("agribrain", "AGRI-BRAIN", G.COLORS["agribrain"]),
             ("pirag_only", "piRAG only", G.COLORS["pirag_only"]),
             ("mcp_only", "MCP only", G.COLORS["mcp_only"])]
    qmat, qse = {}, {}
    for s in SCEN5:
        nb = SUMM.get(s, {}).get("no_context", {}).get("ari", {})
        muB = nb.get("mean")
        if muB is None:
            continue
        q, se = {}, {}
        for mode, _, _ in MODES:
            m = SUMM.get(s, {}).get(mode, {}); ab = m.get("ari", {})
            muA = ab.get("mean"); ib = m.get("context_influenced_steps", {})
            muI = ib.get("mean") if isinstance(ib, dict) else ib
            if muA is None or muI is None:
                continue
            infl = float(muI) / 288.0
            if infl <= 0:
                continue
            dari = float(muA) - float(muB); dq = dari / infl; q[mode] = dq
            try:
                vdiff = ab.get("std", 0.0) ** 2 / max(ab.get("n_seeds", 1), 1) \
                    + nb.get("std", 0.0) ** 2 / max(nb.get("n_seeds", 1), 1)
                vI = (ib.get("std", 0.0) ** 2 / max(ib.get("n_seeds", 1), 1)) if isinstance(ib, dict) else 0.0
                if abs(dari) > 1e-9 and float(muI) > 1e-9:
                    rel = vdiff / dari ** 2 + vI / float(muI) ** 2
                    se[mode] = abs(dq) * float(np.sqrt(max(rel, 0.0)))
                else:
                    se[mode] = 0.0
            except (TypeError, ValueError):
                se[mode] = 0.0
        if q:
            qmat[s] = q; qse[s] = se
    scm = [s for s in SCEN5 if s in qmat]
    x = np.arange(len(scm)); w = 0.8 / len(MODES)
    for i, (mode, lab, col) in enumerate(MODES):
        h = [qmat[s].get(mode, 0.0) for s in scm]
        e = [1.96 * qse[s].get(mode, 0.0) for s in scm]
        axB.bar(x + i * w, h, w, yerr=e if any(e) else None, color=col, label=lab, alpha=0.92,
                edgecolor="white", linewidth=0.7, capsize=4,
                error_kw={"lw": 1.3, "ecolor": "#1F1F1F", "alpha": 0.9} if any(e) else None)
    axB.set_xticks(x + w * (len(MODES) - 1) / 2); axB.set_xticklabels([SLAB[s] for s in scm], rotation=20, ha="right")
    axB.set_ylabel(r"$\Delta$ARI per intervention"); axB.set_title("(b) Context Decision Quality")
    G._apply_style(axB); G._legend(axB, loc="upper right", ncol=1)
    fig.tight_layout(w_pad=2.4)
    _suptitle_flush(fig, (axA, axB), "Performance Gain over Baselines and Context Influence")
    _save(fig, "focapo_evidence"); G._font_restore(saved)


# ===================================================== Fig 3: cross-scenario comparison
def _mv(s, m, metric):
    """(mean, err_low, err_high) for a benchmark_summary cell."""
    d = SUMM.get(s, {}).get(m, {}).get(metric, {})
    if not isinstance(d, dict):
        return (float(d), 0.0, 0.0) if d is not None else (0.0, 0.0, 0.0)
    mean = float(d.get("mean", 0.0) or 0.0)
    lo = d.get("ci_low", mean); hi = d.get("ci_high", mean)
    lo = mean if lo is None else lo; hi = mean if hi is None else hi
    return (mean, max(0.0, mean - lo), max(0.0, hi - mean))


def fig_cross():
    """Cross-scenario performance — ARI / reverse-logistics efficiency / waste /
    SLCA for static, Hybrid RL, integrated across the four disruptions. Rebuilt
    self-contained from benchmark_summary.json (the journal renders the same view
    as generate_figures.fig6_cross)."""
    saved = _bump(FONT_BUMP - 1, panel=30, axis=28)
    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(15, 11.5))
    SC = ["heatwave", "overproduction", "cyber_outage", "adaptive_pricing"]
    modes = ["static", "hybrid_rl", "agribrain"]

    def panel(ax, metric, ylabel, title):
        x = np.arange(len(SC)); w = 0.8 / len(modes)
        for i, m in enumerate(modes):
            vals, elo, ehi = [], [], []
            for s in SC:
                mean, lo, hi = _mv(s, m, metric)
                vals.append(mean); elo.append(lo); ehi.append(hi)
            ax.bar(x + i * w, vals, w, yerr=[elo, ehi], color=G.COLORS[m], label=_lbl(m),
                   alpha=0.92, edgecolor="white", linewidth=0.8, capsize=G._ERR_CAPSIZE, error_kw=G._ERR_KW)
        ax.set_xticks(x + w * (len(modes) - 1) / 2)
        ax.set_xticklabels([SLAB[s] for s in SC], rotation=20, ha="right")
        ax.set_ylabel(ylabel); ax.set_title(title); G._apply_style(ax)

    panel(axA, "ari", "Adaptive Resilience Index", "(a) Resilience Ranking")
    panel(axB, "rle", "Reverse Logistics Efficiency", "(b) Defensive Routing")
    panel(axC, "waste", "Waste Rate", "(c) Waste across Stressors")
    panel(axD, "slca", "SLCA Score", "(d) Sustainability (SLCA)")
    h, l = axA.get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, framealpha=0.9, edgecolor="#757575",
                     fancybox=False, bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_fontweight("bold")
    fig.suptitle("Cross-Scenario Performance Comparison", y=0.975, fontsize=SUP_TITLE, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.955], h_pad=1.6, w_pad=1.4)
    _save(fig, "focapo_cross"); G._font_restore(saved)


# ===================================================== Table 8 data emitter
def emit_table8():
    """Print the 8-mode ablation ARI table (mean per scenario) for the paper."""
    order = [("agribrain", "Integrated"), ("mcp_only", "MCP-only"), ("pirag_only", "piRAG-only"),
             ("no_context", "No-context"), ("no_pinn", "No-PINN"), ("hybrid_rl", "Hybrid-RL"),
             ("no_slca", "No-SLCA"), ("static", "Static control")]
    print("\n=== TABLE 8 (8-mode mean ARI) ===")
    print("%-16s %8s %8s %8s %8s %8s" % ("Mode", "Heatw", "Overpr", "Cyber", "Pricing", "Base"))
    for m, lab in order:
        row = [SUMM.get(s, {}).get(m, {}).get("ari", {}).get("mean") for s in SCEN5]
        print("%-16s " % lab + " ".join(f"{v:8.3f}" if v is not None else "     n/a" for v in row))


if __name__ == "__main__":
    print("rendering conference figures from saved benchmark seeds (no re-simulation)...")
    fig_h1(); fig_cross(); fig_evidence(); emit_table8()
    print("done")
