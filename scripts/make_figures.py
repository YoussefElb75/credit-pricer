"""Generate the README / notebook figures into ``docs/figures`` (light and dark PNGs).

Every figure is rendered twice: ``<name>.png`` on a light surface and
``<name>-dark.png`` on a dark surface, so the README can serve the right one for the
reader's GitHub theme through ``<picture>`` and ``prefers-color-scheme``.

Usage (from the repository root)::

    python scripts/make_figures.py                 # all figures -> docs/figures
    python scripts/make_figures.py --out /tmp/figs

Requires matplotlib; figure 3 (QuantLib validation) also requires QuantLib.
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from credit_pricer.cds import CDS, bucketed_cs01, cs01, price_cds  # noqa: E402
from credit_pricer.cln import CreditLinkedNote, value_cln  # noqa: E402
from credit_pricer.curves import (  # noqa: E402
    DiscountCurve,
    SurvivalCurve,
    bootstrap_survival_curve,
)
from credit_pricer.data import (  # noqa: E402
    VALUATION_DATE,
    load_discount_curve,
    load_index,
    load_single_names,
    load_tranche_quotes,
)
from credit_pricer.index import (  # noqa: E402
    Constituent,
    adjust_to_index_quotes,
    constituents_from_spreads,
    index_intrinsic,
)
from credit_pricer.tranches import (  # noqa: E402
    TranchePricer,
    calibrate_base_correlation,
    finite_pool_default_pmf,
    lhp_loss_cdf,
    lhp_tranche_loss,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "figures"


# ------------------------------------------------------------------------- theme
@dataclass(frozen=True)
class Theme:
    """Colour tokens; the dark theme uses its own validated steps, not an inversion."""

    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str, str]
    seq: str  # colormap for sequential magnitude


LIGHT = Theme(
    "light", "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
    ("#2a78d6", "#eb6834", "#1baf7a"), "Blues",
)
DARK = Theme(
    "dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
    ("#3987e5", "#d95926", "#199e70"), "Blues_r",
)

FONT = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def apply_theme(t: Theme) -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "figure.facecolor": t.surface,
        "axes.facecolor": t.surface,
        "savefig.facecolor": t.surface,
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": 10,
        "text.color": t.ink,
        "axes.labelcolor": t.ink2,
        "axes.titlecolor": t.ink,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.edgecolor": t.axis,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": t.grid,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": t.muted,
        "ytick.color": t.muted,
        "xtick.labelcolor": t.ink2,
        "ytick.labelcolor": t.ink2,
        "legend.frameon": False,
        "legend.labelcolor": t.ink2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
    })


def new_figure(t: Theme, ncols: int = 1, width: float = 8.0, height: float = 3.8,
               **kw: Any) -> tuple[Figure, Any]:
    apply_theme(t)
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), layout="constrained", **kw)
    return fig, axes


def suptitle(fig: Figure, t: Theme, title: str, subtitle: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="semibold", color=t.ink)
    fig.text(0.01, -0.01, subtitle, ha="left", va="top", fontsize=8.5, color=t.muted,
             transform=fig.transFigure)


def marker_kw(t: Theme, color: str, filled: bool = True, size: float = 7) -> dict[str, Any]:
    """Markers carry a surface-coloured ring so they stay legible on lines."""
    return {
        "marker": "o",
        "markersize": size,
        "markerfacecolor": color if filled else t.surface,
        "markeredgecolor": t.surface if filled else color,
        "markeredgewidth": 1.6,
    }


# ------------------------------------------------------------------------- data
class Context:
    """Lazily computed inputs shared by the figures."""

    trade_date: dt.date = VALUATION_DATE

    @cached_property
    def discount(self) -> DiscountCurve:
        return load_discount_curve()

    @cached_property
    def names(self) -> dict[str, tuple[list[int], list[float], SurvivalCurve]]:
        out = {}
        for name, q in load_single_names().items():
            curve = bootstrap_survival_curve(
                self.trade_date, self.discount, q.tenors, q.spreads, q.recovery
            )
            out[name] = (q.tenors, q.spreads, curve)
        return out

    @cached_property
    def index(self) -> Any:
        return load_index()

    @cached_property
    def members(self) -> list[Constituent]:
        d = self.index
        return constituents_from_spreads(
            self.trade_date, self.discount, d.tenors, d.spreads, d.recovery, d.names
        )

    @cached_property
    def adjusted(self) -> Any:
        d = self.index
        return adjust_to_index_quotes(
            self.trade_date, d.quoted_tenors, d.quoted_spreads, d.coupon, self.members,
            self.discount,
        )

    @cached_property
    def tranches(self) -> Any:
        return load_tranche_quotes()

    @cached_property
    def pricer(self) -> TranchePricer:
        curves = [c.curve for c in self.adjusted.constituents]
        return TranchePricer(self.trade_date, self.tranches.tenor, curves, self.discount)

    @cached_property
    def pool_pd_5y(self) -> float:
        return float(self.pricer.default_probs[-1].mean())


def short(name: str) -> str:
    return name.split("_")[0].title()


# ------------------------------------------------------------------------- figures
def fig_hazard_survival(ctx: Context, t: Theme) -> Figure:
    fig, (ax1, ax2) = new_figure(t, 2)
    grid = np.linspace(0.0, 10.0, 1500)
    for color, (name, (_, _, curve)) in zip(t.series, ctx.names.items()):
        h = curve.hazard(grid) * 100
        ax1.step(grid, h, where="post", color=color, label=short(name))
        ax1.annotate(short(name), (grid[-1], h[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=8.5, color=t.ink2)
        q = curve.survival(grid) * 100
        ax2.plot(grid, q, color=color, label=short(name))
        ax2.annotate(f"{short(name)} {q[-1]:.1f}%", (grid[-1], q[-1]), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=8.5, color=t.ink2)
    ax1.set_title("Piecewise-constant hazard rate")
    ax1.set_ylabel("hazard rate (% per year)")
    ax1.set_xlabel("years")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, 12.5)
    ax2.set_title("Survival probability")
    ax2.set_ylabel("Q(t) (%)")
    ax2.set_xlabel("years")
    ax2.set_xlim(0, 13.5)
    suptitle(fig, t, "1. Bootstrapped credit curves",
             "Synthetic par spread quotes, recovery 40%, ISDA conventions.")
    return fig


def fig_round_trip(ctx: Context, t: Theme) -> Figure:
    fig, (ax1, ax2) = new_figure(t, 2, gridspec_kw={"width_ratios": [1.5, 1]})
    for i, (color, (name, (tenors, spreads, curve))) in enumerate(
        zip(t.series, ctx.names.items())
    ):
        fine = np.arange(1, 11)
        rep = [price_cds(CDS.standard(ctx.trade_date, int(k), 0.01), ctx.discount, curve)
               .par_spread * 1e4 for k in fine]
        ax1.plot(fine, rep, color=color, label=f"{short(name)} repriced")
        ax1.plot(tenors, np.array(spreads) * 1e4, linestyle="none",
                 **marker_kw(t, color, filled=False), label=f"{short(name)} quote")
        err = [abs(price_cds(CDS.standard(ctx.trade_date, k, s), ctx.discount, curve)
                   .par_spread - s) * 1e4 for k, s in zip(tenors, spreads)]
        offs = (i - 1) * 0.18
        ax2.plot(np.array(tenors) + offs, np.maximum(err, 1e-16), linestyle="none",
                 **marker_kw(t, color))
    ax1.set_title("Par spread term structure")
    ax1.set_ylabel("par spread (bp)")
    ax1.set_xlabel("maturity (years)")
    ax1.set_yscale("log")
    ticks = [20, 50, 100, 200, 400]
    ax1.set_yticks(ticks, [str(v) for v in ticks])
    ax1.minorticks_off()
    ax1.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax2.set_title("Round-trip error")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-17, 1e-9)
    ax2.set_ylabel("|repriced - quote| (bp)")
    ax2.set_xlabel("maturity (years)")
    suptitle(fig, t, "2. Bootstrap round trip: quotes vs repriced par spreads",
             "Hollow markers: input quotes. Lines: par spreads of standard CDS priced on the "
             "bootstrapped curves (1y to 10y).")
    return fig


def fig_validation(ctx: Context, t: Theme) -> Figure:
    from credit_pricer.validation import (
        VALIDATION_METHODS,
        VALIDATION_SPREADS_BP,
        VALIDATION_TENORS,
        run_validation,
    )

    rows, _ = run_validation(ctx.discount)
    titles = {"isda": "ISDA exact (no bias)", "isda_halfday": "ISDA exact (half-day bias)",
              "midpoint": "Mid-point"}
    fig, axes = new_figure(t, 3, width=9.5, height=3.6, sharey=True)
    floor = 1e-12
    norm = LogNorm(vmin=floor, vmax=1e-3)
    cmap = plt.get_cmap(t.seq)
    image = None
    for ax, method in zip(axes, VALIDATION_METHODS):
        m = np.full((len(VALIDATION_SPREADS_BP), len(VALIDATION_TENORS)), floor)
        for r in rows:
            if r.method == method:
                i = VALIDATION_SPREADS_BP.index(r.spread_bp)
                j = VALIDATION_TENORS.index(r.tenor)
                m[i, j] = max(abs(r.d_upfront_bp), floor)
        image = ax.imshow(m, norm=norm, cmap=cmap, aspect="auto", origin="lower")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                rgb = cmap(norm(m[i, j]))[:3]
                lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
                label = "<1e-12" if m[i, j] <= floor else f"{m[i, j]:.0e}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.5,
                        color="#0b0b0b" if lum > 0.5 else "#ffffff")
        ax.set_title(titles[method], fontsize=10)
        ax.set_xticks(range(len(VALIDATION_TENORS)), [f"{k}y" for k in VALIDATION_TENORS])
        ax.set_yticks(range(len(VALIDATION_SPREADS_BP)),
                      [f"{s:.0f}" for s in VALIDATION_SPREADS_BP])
        ax.grid(False)
        ax.set_xlabel("maturity")
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("par spread level (bp)")
    assert image is not None
    cb = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("|ours - QuantLib| value (bp of notional)", color=t.ink2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=t.muted, labelcolor=t.ink2)
    import QuantLib as ql

    suptitle(fig, t, f"3. Validation against QuantLib {ql.__version__}: absolute value difference",
             "Identical curves in both libraries; standard CDS, 100 bp coupon, recovery 40%.")
    return fig


def fig_cs01(ctx: Context, t: Theme) -> Figure:
    name = "BRAVO_IG_MID"
    tenors, spreads, curve = ctx.names[name]
    cds = CDS.standard(ctx.trade_date, 5, 0.01)
    buckets = bucketed_cs01(cds, ctx.discount, tenors, spreads) * 1e4  # per 10k notional
    parallel = cs01(cds, ctx.discount, tenors, spreads) * 1e4
    rpv01 = price_cds(cds, ctx.discount, curve).rpv01_bp * 1e4
    fig, ax = new_figure(t, 1, width=7.0, height=3.6)
    x = np.arange(len(tenors))
    bars = ax.bar(x, buckets, width=0.6, color=t.series[0], edgecolor=t.surface, linewidth=2)
    for b, v in zip(bars, buckets):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, max(v, 0)), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8.5, color=t.ink2)
    ax.set_xticks(x, [f"{k}y" for k in tenors])
    ax.set_xlabel("quoted tenor bumped by +1 bp")
    ax.set_ylabel("CS01 (per 10,000 notional)")
    ax.set_title(f"Protection buyer, 5y CDS on {short(name)}, 100 bp coupon")
    ax.text(0.02, 0.92, f"parallel CS01 {parallel:.3f}   sum of buckets {buckets.sum():.3f}"
            f"   RPV01 x 1 bp {rpv01:.3f}", transform=ax.transAxes, fontsize=8.5,
            color=t.ink2)
    ax.set_ylim(min(0.0, float(np.min(buckets))) * 1.2, float(np.max(buckets)) * 1.25)
    ax.grid(axis="x", visible=False)
    suptitle(fig, t, "4. Bucketed CS01",
             "Curve rebuilt after each bump. The 5y quote carries almost all the risk.")
    return fig


def fig_index(ctx: Context, t: Theme) -> Figure:
    d = ctx.index
    intr = [index_intrinsic(ctx.trade_date, k, d.coupon, ctx.members, ctx.discount).spread * 1e4
            for k in d.quoted_tenors]
    quoted = np.array(d.quoted_spreads) * 1e4
    basis = quoted - np.array(intr)
    fig, (ax1, ax2) = new_figure(t, 2, gridspec_kw={"width_ratios": [1.4, 1]})
    ax1.plot(d.quoted_tenors, intr, color=t.series[0], **marker_kw(t, t.series[0]),
             label="intrinsic (125 constituents)")
    ax1.plot(d.quoted_tenors, quoted, color=t.series[1], **marker_kw(t, t.series[1]),
             label="quoted index spread")
    ax1.set_title("Index spread term structure")
    ax1.set_xlabel("maturity (years)")
    ax1.set_ylabel("spread (bp)")
    ax1.legend(loc="lower right")
    ax1.set_xticks(d.quoted_tenors)
    bars = ax2.bar(range(len(basis)), basis, width=0.6, color=t.series[1],
                   edgecolor=t.surface, linewidth=2)
    for b, v in zip(bars, basis):
        ax2.annotate(f"{v:+.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, -11),
                     textcoords="offset points", ha="center", fontsize=8.5, color=t.ink2)
    ax2.axhline(0, color=t.axis, linewidth=1)
    ax2.set_xticks(range(len(basis)), [f"{k}y" for k in d.quoted_tenors])
    ax2.set_title("Index basis (quoted - intrinsic)")
    ax2.set_ylabel("bp")
    ax2.set_ylim(basis.min() * 1.35, 0.5)
    ax2.grid(axis="x", visible=False)
    f = ", ".join(f"{k}y {a:.3f}" for k, a in zip(ctx.adjusted.tenors, ctx.adjusted.factors))
    suptitle(fig, t, "5. CDS index: intrinsic vs quoted spread",
             f"Hazard multipliers that close the basis by maturity segment: {f}.")
    return fig


def fig_loss_distribution(ctx: Context, t: Theme) -> Figure:
    n, r = 125, 0.4
    p = ctx.pool_pd_5y
    lgd_unit = (1 - r) / n
    fig, axes = new_figure(t, 3, width=9.5, height=3.5, sharey=False)
    for ax, rho in zip(axes, (0.1, 0.3, 0.6)):
        pmf = finite_pool_default_pmf(np.full(n, p), rho)
        k = np.arange(n + 1)
        loss = k * lgd_unit * 100
        keep = loss <= 15
        # LHP probability of the same loss buckets, for a like-for-like comparison
        edges = (np.arange(n + 2) - 0.5) * lgd_unit
        lhp = np.diff(lhp_loss_cdf(edges, p, r, rho))
        ax.bar(loss[keep], pmf[keep] * 100, width=lgd_unit * 100 * 0.8,
               color=t.series[0], label=f"finite pool, {n} names")
        ax.plot(loss[keep], lhp[keep] * 100, color=t.series[1], label="LHP")
        ax.set_title(f"rho = {rho:.1f}")
        ax.set_xlabel("portfolio loss (% of notional)")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("probability (%)")
    axes[0].legend(loc="upper right")
    suptitle(fig, t, "6. 5y portfolio loss distribution: LHP vs finite-pool recursion",
             f"Homogeneous pool, 5y default probability {p:.2%} (synthetic index average), "
             "recovery 40%. LHP mass binned on the same loss grid.")
    return fig


def fig_tranche_el(ctx: Context, t: Theme) -> Figure:
    p, r = ctx.pool_pd_5y, 0.4
    rhos = np.linspace(0.0, 0.99, 100)
    tranches = [(0.0, 0.03, "equity 0-3%"), (0.03, 0.07, "mezzanine 3-7%"),
                (0.15, 0.30, "senior 15-30%")]
    fig, ax = new_figure(t, 1, width=7.5, height=3.8)
    for color, (a, d, label) in zip(t.series, tranches):
        el = [lhp_tranche_loss(a, d, p, r, rho) / (d - a) * 100 for rho in rhos]
        ax.plot(rhos, el, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_xlabel("flat correlation rho")
    ax.set_ylabel("5y expected loss (% of tranche)")
    ax.set_title("Expected tranche loss vs correlation (LHP)")
    ax.legend(loc="upper center")
    suptitle(fig, t, "7. Correlation sensitivity: equity is long correlation, senior is short",
             f"Pool 5y default probability {p:.2%}, recovery 40%. Higher correlation moves "
             "losses from the equity tranche to the senior tranche.")
    return fig


def fig_base_correlation(ctx: Context, t: Theme) -> Figure:
    tr = ctx.tranches
    curve = calibrate_base_correlation(ctx.pricer, tr.quotes)
    k = np.array([pt.detach for pt in curve]) * 100
    got = np.array([pt.correlation for pt in curve])
    gen = np.array(tr.generating_correlations)
    err = np.abs(got - gen).max()
    fig, ax = new_figure(t, 1, width=7.0, height=3.8)
    ax.plot(k, gen, linestyle="none", **marker_kw(t, t.series[1], filled=False, size=12),
            label="generating skew (input)")
    ax.plot(k, got, color=t.series[0], **marker_kw(t, t.series[0]),
            label="calibrated from tranche quotes")
    for x, y in zip(k, got):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(6, -12), textcoords="offset points",
                    fontsize=8.5, color=t.ink2)
    ax.set_xlabel("detachment point (%)")
    ax.set_ylabel("base correlation")
    ax.set_xticks(k)
    ax.set_ylim(0, 1)
    ax.set_title(f"Round-trip max |error| = {err:.1e}")
    ax.legend(loc="upper left")
    suptitle(fig, t, "8. Base correlation curve (5y, synthetic tranche quotes)",
             "Finite-pool Gaussian copula on the 125 index-adjusted curves; quotes are "
             "upfront + running coupon.")
    return fig


def fig_cln(ctx: Context, t: Theme) -> Figure:
    levels = np.array([20, 50, 100, 200, 350, 500, 750, 1000]) * 1e-4
    fundings = [(0.0, "funding 0 bp"), (0.005, "funding 50 bp"), (0.015, "funding 150 bp")]
    note = CreditLinkedNote.standard(ctx.trade_date, 5, 0.0)
    fig, (ax1, ax2) = new_figure(t, 2, gridspec_kw={"width_ratios": [1.3, 1]})
    hi = 0.0
    for color, (f, label) in zip(t.series, fundings):
        xs, ys = [], []
        for s in levels:
            curve = bootstrap_survival_curve(ctx.trade_date, ctx.discount, [1, 3, 5, 7],
                                             [s] * 4)
            par = price_cds(CDS.standard(ctx.trade_date, 5, 0.01), ctx.discount, curve)
            fair = value_cln(note, ctx.discount, curve, f).fair_spread
            xs.append((par.par_spread + f * 360 / 365) * 1e4)
            ys.append(fair * 1e4)
        xs_a, ys_a = np.array(xs), np.array(ys)
        hi = max(hi, xs_a.max(), ys_a.max())
        ax1.plot(xs_a, ys_a, linestyle="none", **marker_kw(t, color), label=label)
        ax2.plot(xs_a, ys_a - xs_a, color=color, **marker_kw(t, color), label=label)
    ax1.plot([0, hi * 1.05], [0, hi * 1.05], color=t.muted, linewidth=1, zorder=0)
    ax1.annotate("y = x", (hi * 0.8, hi * 0.8), xytext=(-24, 6), textcoords="offset points",
                 fontsize=8.5, color=t.muted)
    ax1.set_xlabel("CDS par spread + funding spread (bp)")
    ax1.set_ylabel("CLN fair spread (bp)")
    ax1.set_title("CLN fair spread vs replication")
    ax1.legend(loc="upper left")
    ax2.axhline(0, color=t.axis, linewidth=1)
    ax2.set_xlabel("CDS par spread + funding spread (bp)")
    ax2.set_ylabel("difference (bp)")
    ax2.set_title("CLN - (CDS + funding)")
    suptitle(fig, t, "9. Credit-linked note = funded CDS + issuer funding",
             "5y floating-rate CLN, recovery 40%, flat synthetic curves. The gap is the accrued "
             "coupon the CLN forfeits at default (the CDS pays accrual on default).")
    return fig


FIGURES: dict[str, Callable[[Context, Theme], Figure]] = {
    "01_hazard_survival": fig_hazard_survival,
    "02_spread_round_trip": fig_round_trip,
    "03_quantlib_validation": fig_validation,
    "04_cs01_buckets": fig_cs01,
    "05_index_basis": fig_index,
    "06_loss_distribution": fig_loss_distribution,
    "07_tranche_el_vs_correlation": fig_tranche_el,
    "08_base_correlation": fig_base_correlation,
    "09_cln_spread": fig_cln,
}


def render(name: str, ctx: Context, out: Path, themes: tuple[Theme, ...] = (LIGHT, DARK)) -> None:
    for t in themes:
        fig = FIGURES[name](ctx, t)
        suffix = "" if t.name == "light" else "-dark"
        fig.savefig(out / f"{name}{suffix}.png", dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ctx = Context()
    for name in args.only or FIGURES:
        render(name, ctx, args.out)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
