
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

CSV_FILE = "radar_targets.csv"
OUTPUT_FILE = "figure_radar_roi.png"

X_MIN = -1000
X_MAX = 1000
Y_MIN = 600
Y_MAX = 2100

PLOT_X_MIN = -3000
PLOT_X_MAX = 3000
PLOT_Y_MIN = -500
PLOT_Y_MAX = 6000

STAGES = ["EMPTY", "SITTING", "LYING"]

df = pd.read_csv(CSV_FILE)

df = df[
    (df["detected"] == 1) &
    (df["stage"].isin(STAGES))
].copy()

df["x_mm"] = pd.to_numeric(df["x_mm"], errors="coerce")
df["y_mm"] = pd.to_numeric(df["y_mm"], errors="coerce")

df = df.dropna(subset=["x_mm", "y_mm"])

df["inside_roi"] = (
    df["x_mm"].between(X_MIN, X_MAX) &
    df["y_mm"].between(Y_MIN, Y_MAX)
)

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5),
    sharex=True,
    sharey=True
)

for ax, stage in zip(axes, STAGES):

    stage_df = df[df["stage"] == stage]


    inside = stage_df[stage_df["inside_roi"]]
    outside = stage_df[~stage_df["inside_roi"]]

    ax.scatter(
        outside["x_mm"],
        outside["y_mm"],
        s=8,
        c="darkorange",
        alpha=0.35,
        edgecolors="none",
        rasterized=True
    )

    ax.scatter(
        inside["x_mm"],
        inside["y_mm"],
        s=8,
        c="green",
        alpha=0.35,
        edgecolors="none",
        rasterized=True
    )

    roi = Rectangle(
        (X_MIN, Y_MIN),
        X_MAX - X_MIN,
        Y_MAX - Y_MIN,
        fill=False,
        edgecolor="green",
        linewidth=2,
        linestyle="--"
    )

    ax.add_patch(roi)

    ax.set_title(stage, fontsize=15, fontweight="bold")

    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)

    ax.set_xlabel("X Position (mm)")
    ax.set_ylabel("Y Position (mm)")

    ax.grid(True, alpha=0.15)

    ax.set_aspect("equal", adjustable="box")

legend_items = [
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="green",
        markersize=7,
        label="Inside Bed ROI"
    ),
    Line2D(
        [0], [0],
        marker="o",
        color="w",
        markerfacecolor="darkorange",
        markersize=7,
        label="Outside Bed ROI"
    ),
    Line2D(
        [0], [0],
        color="green",
        linestyle="--",
        linewidth=2,
        label="Bed ROI"
    )
]

fig.legend(
    handles=legend_items,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0.09, 1, 1])

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {OUTPUT_FILE}")
