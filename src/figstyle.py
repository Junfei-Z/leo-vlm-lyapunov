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
    mpl.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 16,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 2.0,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,   # embed TrueType so text stays selectable
        "ps.fonttype": 42,
    })

def savefig_pub(fig, path_pdf):
    fig.tight_layout(pad=2)
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")
