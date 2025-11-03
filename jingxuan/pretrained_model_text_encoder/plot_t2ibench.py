# This cell creates a pre-filled CSV template and helper functions to make your plots.
# You can download the CSV, fill in the numbers from your table, upload it back here,
# and I'll run the plotting functions to generate the figures.

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from textwrap import wrap
import os
import argparse

# Try to import the optional UI helper; continue gracefully if unavailable
try:
    from ace_tools import display_dataframe_to_user  # type: ignore
except Exception:  # noqa: BLE001
    display_dataframe_to_user = None

# 1) Create a CSV template with model names and encoder types pre-filled
models = [
    "Stable v1-4",
    "Stable v2",
    "Composable + SD v2",
    "Structured + SD v2",
    "Attn-Exct + SD v2",
    "GORS-unbiased + SD v2 (ours)",
    "GORS + SD v2 (ours)",
    "Stable XL",
    "Pixart-α-ft",
    "DALL·E 3",
]

encoder_map = {
    "Stable v1-4": "CLIP ViT-L/14",
    "Stable v2": "OpenCLIP ViT-H/14",
    "Composable + SD v2": "OpenCLIP ViT-H/14",
    "Structured + SD v2": "OpenCLIP ViT-H/14",
    "Attn-Exct + SD v2": "OpenCLIP ViT-H/14",
    "GORS-unbiased + SD v2 (ours)": "OpenCLIP ViT-H/14",
    "GORS + SD v2 (ours)": "OpenCLIP ViT-H/14",
    "Stable XL": "OpenCLIP ViT-bigG/14 + CLIP ViT-L/14",
    "Pixart-α-ft": "T5 (t5-v1_1-xxl)",
    "DALL·E 3": "Unknown",
}

df_template = pd.DataFrame({
    "model": models,
    "encoder_type": [encoder_map[m] for m in models],
    # Fill these numeric columns with the values from your table:
    "color": np.nan,
    "shape": np.nan,
    "texture": np.nan,
    "spatial_2d": np.nan,
    "spatial_3d": np.nan,
})


def create_csv_template(path: str = "T2I-CompBench_subset.csv", show: bool = True) -> str:
    """Write a pre-filled CSV template and optionally display it if UI helper is available."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_template.to_csv(csv_path, index=False)
    if show and display_dataframe_to_user is not None:
        display_dataframe_to_user("T2I-CompBench_subset template", df_template)
    print(f"Saved a fill-in template to: {csv_path}")
    return str(csv_path)


def _simplify_encoder_group(enc):
    """Map detailed encoders to a small set of legend-friendly groups."""
    if enc is None or pd.isna(enc):
        return "Unknown"
    enc = str(enc)
    if "bigG" in enc or ("+" in enc and "CLIP" in enc and "OpenCLIP" in enc):
        return "Dual (OpenCLIP bigG + CLIP L/14)"
    if enc.startswith("OpenCLIP") or "OpenCLIP" in enc:
        if "H/14" in enc:
            return "OpenCLIP ViT-H/14"
        return "OpenCLIP (other)"
    if enc.startswith("CLIP") or ("CLIP" in enc and "OpenCLIP" not in enc):
        return "CLIP ViT-L/14"
    if enc.startswith("T5") or "t5" in enc.lower():
        return "T5 (text)"
    return "Unknown"


def _palette():
    """A fixed palette for encoder groups (you asked to color by text encoder)."""
    return {
        "CLIP ViT-L/14": "#1f77b4",               # blue
        "OpenCLIP ViT-H/14": "#ff7f0e",           # orange
        "Dual (OpenCLIP bigG + CLIP L/14)": "#2ca02c",  # green
        "T5 (text)": "#d62728",                    # red
        "OpenCLIP (other)": "#9467bd",            # purple
        "Unknown": "#7f7f7f",                      # gray
    }


def load_and_prepare(path):
    df = pd.read_csv(path)
    # Compute aggregates
    df["single_mean"] = df[["color", "shape", "texture"]].mean(axis=1, skipna=True)
    df["spatial_mean"] = df[["spatial_2d", "spatial_3d"]].mean(axis=1, skipna=True)
    df["diff_single_minus_spatial"] = df["single_mean"] - df["spatial_mean"]
    df["encoder_group"] = df["encoder_type"].apply(_simplify_encoder_group)
    # Drop rows where both aggregates are missing
    df = df.dropna(subset=["single_mean", "spatial_mean"], how="any")
    return df


def plot_dumbbell(df, out_path="plots/dumbbell_single_vs_spatial.pdf"):
    # Sort by difference descending
    plot_df = df.sort_values("diff_single_minus_spatial", ascending=False).reset_index(drop=True)
    enc_colors = _palette()
    fig = plt.figure(figsize=(10, max(5, 0.5 * len(plot_df) + 2)))
    y_positions = np.arange(len(plot_df))

    for i, row in plot_df.iterrows():
        c = enc_colors.get(row["encoder_group"], "#7f7f7f")
        # line from spatial_mean to single_mean
        x0, x1 = row["spatial_mean"], row["single_mean"]
        plt.plot([x0, x1], [i, i], linewidth=2, color=c, alpha=0.8)
        plt.scatter([x0, x1], [i, i], s=40, color=c, zorder=3)

    plt.yticks(y_positions, [ "\n".join(wrap(m, 28)) for m in plot_df["model"] ])
    plt.xlabel("Mean accuracy")
    plt.title("Single-object properties vs. Spatial relationships (per model)")
    # Add legend for encoders
    used_groups = list(dict.fromkeys(plot_df["encoder_group"]))  # preserve order
    handles = [plt.Line2D([0],[0], color=_palette().get(g, "#7f7f7f"), lw=3) for g in used_groups]
    plt.legend(handles, used_groups, title="Text encoder", frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_scatter(df, out_path="plots/scatter_single_vs_spatial.pdf"):
    enc_colors = _palette()
    fig = plt.figure(figsize=(10, 8))  # Increased figure size for more space
    
    # diagonal
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1, color="#aaaaaa")
    
    # First pass: plot points and collect positions
    points = []
    labels = []
    colors = []
    
    for _, row in df.iterrows():
        c = enc_colors.get(row["encoder_group"], "#7f7f7f")
        x, y = row["single_mean"], row["spatial_mean"]
        plt.scatter(x, y, s=80, color=c, edgecolors='white', linewidth=0.5, zorder=3)  # Slightly larger points with edge
        points.append((x, y))
        
        # Create shorter labels for better readability
        model_name = row["model"]
        if "GORS" in model_name and "ours" in model_name:
            if "unbiased" in model_name:
                short_name = "GORS-unbiased (ours)"
            else:
                short_name = "GORS (ours)"
        elif "Stable" in model_name:
            short_name = model_name.replace("Stable", "SD")
        elif "DALL·E" in model_name:
            short_name = "DALL-E 3"
        else:
            short_name = model_name
            
        labels.append(short_name)
        colors.append(c)
    
    # Smart label placement to avoid overlap
    def place_labels_smart(points, labels, colors):
        """Place labels with collision avoidance using longer offsets"""
        label_positions = []
        used_positions = set()
        
        for i, ((x, y), label, color) in enumerate(zip(points, labels, colors)):
            # Special positioning for specific models
            if "Composable" in label and "SD" in label:
                # Force lower left positioning for Composable + SD v2
                offsets = [(-0.12, -0.08), (-0.10, -0.10), (-0.08, -0.12)]
            elif "Attn-Exct" in label and "SD" in label:
                # Force horizontal right positioning for Attn-Exct + SD v2
                offsets = [(0.12, 0.00), (0.10, 0.02), (0.08, -0.02)]
            elif "Structured" in label and "SD" in label:
                # Force vertical down positioning for Structured + SD v2
                offsets = [(0.00, -0.12), (0.02, -0.10), (-0.02, -0.08)]
            elif "GORS-unbiased" in label and "SD" in label:
                # Force vertical up positioning for GORS-unbiased + SD v2
                offsets = [(0.00, 0.12), (0.02, 0.10), (-0.02, 0.08)]
            elif "Pixart" in label:
                # Force vertical up positioning for Pixart-α-ft
                offsets = [(0.00, 0.12), (0.02, 0.10), (-0.02, 0.08)]
            elif ("SD" in label and "v1-4" in label) or ("SD" in label and "v2" in label and "GORS" not in label and "Composable" not in label and "Structured" not in label and "Attn-Exct" not in label):
                # Force upper left positioning for SD v1-4 and SD v2
                offsets = [(-0.12, 0.08), (-0.10, 0.10), (-0.08, 0.12)]
            elif "XL" in label:
                # Force upper left positioning for Stable XL
                offsets = [(-0.12, 0.08), (-0.10, 0.10), (-0.08, 0.12)]
            elif "GORS" in label and "SD" in label and "unbiased" not in label:
                # Force horizontal right positioning for GORS + SD v2 (not unbiased)
                offsets = [(0.12, 0.00), (0.10, 0.02), (0.08, -0.02)]
            else:
                # Use longer offset positions to spread labels further out
                offsets = [
                    (0.08, 0.08),    # top-right (longer)
                    (0.08, -0.08),   # bottom-right (longer)
                    (-0.08, 0.08),   # top-left (longer)
                    (-0.08, -0.08),  # bottom-left (longer)
                    (0.12, 0.02),    # far right
                    (-0.12, 0.02),   # far left
                    (0.02, 0.12),    # far top
                    (0.02, -0.12),   # far bottom
                    (0.10, 0.05),    # top-right medium
                    (0.10, -0.05),   # bottom-right medium
                    (-0.10, 0.05),   # top-left medium
                    (-0.10, -0.05),  # bottom-left medium
                ]
            
            best_offset = offsets[0]  # default
            min_conflicts = float('inf')
            
            for offset_x, offset_y in offsets:
                label_x = x + offset_x
                label_y = y + offset_y
                
                # Check bounds with more generous margins
                if label_x < -0.03 or label_x > 1.08 or label_y < -0.03 or label_y > 1.08:
                    continue
                
                # Count conflicts with existing labels
                conflicts = 0
                for prev_x, prev_y in used_positions:
                    dist = ((label_x - prev_x)**2 + (label_y - prev_y)**2)**0.5
                    if dist < 0.06:  # Reduced minimum distance threshold since we have more space
                        conflicts += 1
                
                if conflicts < min_conflicts:
                    min_conflicts = conflicts
                    best_offset = (offset_x, offset_y)
                    if conflicts == 0:
                        break
            
            final_x = x + best_offset[0]
            final_y = y + best_offset[1]
            
            # Ensure label stays within expanded bounds
            final_x = max(-0.02, min(1.07, final_x))
            final_y = max(-0.02, min(1.07, final_y))
            
            label_positions.append((final_x, final_y))
            used_positions.add((final_x, final_y))
            
            # Always draw connecting line since we're using longer offsets
            plt.plot([x, final_x], [y, final_y], color=color, alpha=0.4, linewidth=1.0, zorder=1)
            
            # Place the label
            plt.text(final_x, final_y, label, fontsize=9, color=color, 
                    weight='bold', ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=color, alpha=0.9, linewidth=0.7))
    
    place_labels_smart(points, labels, colors)
    
    plt.xlabel("Mean(single: color, shape, texture)", fontsize=12)
    plt.ylabel("Mean(spatial: 2D, 3D)", fontsize=12)
    plt.title("Across models: single-object vs spatial", fontsize=14, pad=20)
    plt.xlim(-0.15, 1.15)  # Expanded limits to accommodate longer label offsets
    plt.ylim(-0.15, 1.15)
    
    # Add legend
    used_groups = list(dict.fromkeys([_simplify_encoder_group(row["encoder_type"]) for _, row in df.iterrows()]))
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=_palette().get(g, "#7f7f7f"), 
                         markersize=8, markeredgecolor='white') for g in used_groups]
    plt.legend(handles, used_groups, title="Text encoder", frameon=True, 
              loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_diff_bars(df, out_path="plots/diff_bars.pdf"):
    plot_df = df.sort_values("diff_single_minus_spatial", ascending=False).reset_index(drop=True)
    enc_colors = _palette()
    fig = plt.figure(figsize=(9, max(5, 0.5 * len(plot_df) + 2)))
    y_positions = np.arange(len(plot_df))
    bar_colors = [enc_colors.get(g, "#7f7f7f") for g in plot_df["encoder_group"]]
    plt.barh(y_positions, plot_df["diff_single_minus_spatial"], color=bar_colors)
    plt.yticks(y_positions, [ "\n".join(wrap(m, 28)) for m in plot_df["model"] ])
    plt.gca().invert_yaxis()
    plt.xlabel("Δ (single - spatial)")
    plt.title("Advantage of single-object properties over spatial relationships")
    plt.axvline(0, color="#333333", linewidth=1)
    # Legend
    used_groups = list(dict.fromkeys(plot_df["encoder_group"]))
    handles = [plt.Rectangle((0,0),1,1, color=_palette().get(g, "#7f7f7f")) for g in used_groups]
    plt.legend(handles, used_groups, title="Text encoder", frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_group_boxplot(df, out_path="plots/boxplot_group_summary.pdf"):
    # Boxplot comparing the two groups across models
    singles = df["single_mean"].dropna().values
    spatials = df["spatial_mean"].dropna().values
    fig = plt.figure(figsize=(6,6))
    bp = plt.boxplot([singles, spatials], labels=["Single (C/S/T)", "Spatial (2D/3D)"], showmeans=True)
    plt.ylabel("Mean accuracy")
    plt.title("Summary across models")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_plots_from_csv(path="T2I-CompBench_subset.csv", outdir="plots"):
    df = load_and_prepare(path)
    if df.empty:
        print("No rows with complete aggregates yet. Fill in the CSV first.")
        return []
    
    # Ensure output directory exists
    Path(outdir).mkdir(parents=True, exist_ok=True)
    
    paths = [
        plot_dumbbell(df, out_path=str(Path(outdir) / "dumbbell_single_vs_spatial.pdf")),
        plot_scatter(df, out_path=str(Path(outdir) / "scatter_single_vs_spatial.pdf")),
        plot_diff_bars(df, out_path=str(Path(outdir) / "diff_bars.pdf")),
        plot_group_boxplot(df, out_path=str(Path(outdir) / "boxplot_group_summary.pdf")),
    ]
    return paths


def main():
    parser = argparse.ArgumentParser(description="T2I-CompBench plotting helper")
    action = parser.add_mutually_exclusive_group(required=False)
    action.add_argument("--template", action="store_true", help="Create CSV template and exit")
    action.add_argument("--plots", action="store_true", help="Generate plots from a CSV file and exit")
    parser.add_argument("--csv", default="T2I-CompBench_subset.csv", help="Path to CSV for reading/writing")
    parser.add_argument("--outdir", default="plots", help="Directory to write plots when --plots is used")
    parser.add_argument("--no-show", action="store_true", help="Do not try to display the template via UI helper")

    args = parser.parse_args()

    if args.template:
        create_csv_template(args.csv, show=(not args.no_show))
        return

    if args.plots:
        outputs = make_plots_from_csv(args.csv, args.outdir)
        if outputs:
            print("Wrote:")
            for p in outputs:
                print(f" - {p}")
        return

    # If no action specified, show a short help message
    parser.print_help()


if __name__ == "__main__":
    main()


