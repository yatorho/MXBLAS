from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

fontsize = 18

file_path = "bench_all.csv"
df = pd.read_csv(file_path)

sp_values_to_plot = ["BB", "GB", "CC", "TT"]
sp_values_to_plot_print = ["BxB", "GxB", "CxC", "TxT"]
df_filtered = df[df["SP"].isin(sp_values_to_plot)].copy()

config_cols = [
    "M",
    "N",
    "K",
    "SM",
    "SN",
    "SK",
    "Q",
    "QN",
    "SP",
]  # SP is now part of the config
mxblas_tflops = (
    df_filtered[df_filtered["Method"] == "MXBLAS"].groupby(config_cols)["TFLOPS"].mean()
)


def get_mxblas_tflops(row, tflops_map):
    config_tuple = tuple(row[col] for col in config_cols)
    return tflops_map.get(config_tuple, None)


df_filtered["MXBLAS_TFLOPS_Ref"] = df_filtered.apply(
    lambda row: get_mxblas_tflops(row, mxblas_tflops), axis=1  # type: ignore
)

df_plot = df_filtered

fig, axes = plt.subplots(
    2, 2, figsize=(24, 10), sharey=False
)  # Share Y axis for easier comparison
axes = axes.flatten()  # Flatten the 2x2 array into a 1D array for easy iteration

global_handles_labels = OrderedDict()


unique_methods = df_plot["Method"].unique()
viridis_colors = [
    "#041e42",
    "#b2cae4",
    "#bab49e",
    "#b26801",
    "#008080",
    "#d4af37",
    "#ff7f50",
]
method_colors = dict(zip(unique_methods, viridis_colors))


for i, sp_val in enumerate(sp_values_to_plot):
    ax = axes[i]
    df_subset = df_plot[df_plot["SP"] == sp_val]

    if df_subset.empty:
        ax.text(
            0.5,
            0.5,
            f"No data for SP={sp_val}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )
        ax.set_xlabel("M Dimension")
        ax.set_ylabel("TFLOPS")

    methods_in_subset = df_subset["Method"].unique().tolist()
    if "MXBLAS" in methods_in_subset:
        methods_in_subset.remove("MXBLAS")
        methods_in_subset.sort()  # Sort other methods alphabetically (or however you prefer)
        methods_in_subset.append("MXBLAS")  # Add MXBLAS at the end
    else:
        methods_in_subset.sort()  # Sort if MXBLAS wasn't even in the filtered data

    sns.boxplot(
        data=df_subset,
        x="M",
        y="TFLOPS",
        hue="Method",
        width=0.7,
        patch_artist=True,
        hue_order=methods_in_subset,
        ax=ax,  # Specify the axis to plot on
        palette=method_colors,  # Pass the color mapping here
    )

    # --- Collect handles/labels for Global Legend ---
    handles, labels = ax.get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label not in global_handles_labels:  # Add only if label is unique
            global_handles_labels[label] = handle

    # -------------------------------------------------
    # Customize subplot
    ax.set_title(f"SP = {sp_values_to_plot_print[i]}", fontsize=fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xlabel("M-Dimension", fontsize=fontsize)
    ax.legend(
        handles=handles,
        fontsize=fontsize - 4,
    )

    ax.set_ylabel("TFLOPS", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize - 4)
    ax.tick_params(axis="y", labelsize=fontsize - 4)

sorted_handles = list(global_handles_labels.values())
sorted_labels = list(global_handles_labels.keys())

# 7. Adjust Layout and Display
plt.tight_layout(
    rect=(0, 0.03, 0.95, 0.93), w_pad=5.0
)  # Adjust rect to make space for suptitle and legend
plt.savefig("figure9_boxplot.jpg", dpi=100, bbox_inches="tight")
plt.show()
