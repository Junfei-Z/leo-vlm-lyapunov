import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "blue_main": "#0F4D92",   # the PROPOSED method (Lyapunov / ours)
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",  # greedy-quality / unsafe comparators
    "neutral": "#767676",
    "highlight": "#FFD700",
}

def apply_house_style():
    # Big Times New Roman text that stays legible when figures are shrunk into the
    # 2-column layout, thick lines, distinct markers (the user's line-plot preference,
    # sized so it also works for the multi-panel sensitivity/comparison grids).
    mpl.rcParams.update({
        "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "font.size": 20,
        "axes.labelsize": 24,
        "axes.titlesize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 19,
        "legend.fontsize": 18,
        "lines.linewidth": 3.0,
        "lines.markersize": 9,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,   # embed TrueType so text stays selectable
        "ps.fonttype": 42,
    })

def savefig_pub(fig, path_pdf):
    fig.tight_layout(pad=2)
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")


# =============================================================================
# Line-plot house style (user-specified): big Times New Roman text that stays
# legible after the figure is shrunk into a 2-column layout, thick lines, and
# high-contrast per-series marker+linestyle so curves are distinguishable in B/W.
# =============================================================================
def apply_lineplot_style():
    mpl.rcParams.update({
        "font.family": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "axes.labelsize": 45,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "legend.fontsize": 40,
        "lines.linewidth": 3,
        "figure.figsize": (17.68, 14.48),
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# Stable (color, marker, linestyle) PER POLICY so a curve looks identical across
# every figure. Proposed is the most prominent (solid). markersize set at plot time.
POLICY_STYLE = {
    "Proposed":        dict(color="red",    marker="v", linestyle="-"),
    "Lyapunov (ours)": dict(color="red",    marker="v", linestyle="-"),
    "Greedy-Q":        dict(color="blue",   marker="o", linestyle=":"),
    "Greedy-E":        dict(color="green",  marker="*", linestyle="-."),
    "Random":          dict(color="orange", marker="s", linestyle=(0, (5, 5))),
    "Static":          dict(color="black",  marker="^", linestyle=(0, (3, 1, 1, 1))),
    "MILP":            dict(color="red",    marker="v", linestyle="-"),  # offline optimum
}
# Fallback cycle for non-policy series (user's example order).
LINE_STYLES = [
    dict(color="red",    marker="v", linestyle="-"),
    dict(color="blue",   marker="o", linestyle=":"),
    dict(color="green",  marker="*", linestyle="-."),
    dict(color="black",  marker="^", linestyle=(0, (3, 1, 1, 1))),
    dict(color="orange", marker="s", linestyle=(0, (5, 5))),
]

def plot_series(ax, x, y, label, markersize=22, **kw):
    """Plot one labelled curve using its stable per-policy style."""
    st = dict(POLICY_STYLE.get(label, LINE_STYLES[0])); st.update(kw)
    return ax.plot(x, y, label=label, markersize=markersize, **st)

def finalize_lineplot(ax, xlabel, ylabel, legend_loc="best", y_headroom=0.10):
    """Grid + labels + legend, plus a y-headroom pad so curves don't touch the
    frame and the legend has room (helps avoid legend/line overlap). Caller still
    sets explicit x/y ticks and checks the legend does not cover the data."""
    ax.grid(linestyle="-.")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_headroom:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * y_headroom)
    ax.legend(loc=legend_loc)

def savefig_lines(fig, path_pdf):
    """Save a line figure as vector PDF (+ matching .eps for the user's pipeline)."""
    fig.tight_layout()
    fig.savefig(path_pdf, dpi=1000, bbox_inches="tight")
    if path_pdf.endswith(".pdf"):
        fig.savefig(path_pdf[:-4] + ".eps", dpi=1000, bbox_inches="tight")
