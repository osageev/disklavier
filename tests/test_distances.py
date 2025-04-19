import os
import sys
import faiss
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math  # For sqrt
from rich.progress import Progress

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.utils import basename

# --- configuration ---
FAISS_INDEX_PATH = "/media/scratch/sageev-midi/20250410/specdiff.faiss"
DATA_DIR = "/media/scratch/sageev-midi/20250410/augmented"
TRACK_DIR = "/media/scratch/sageev-midi/20250410/unsegmented"
OUTPUT_DIR = "tests/outputs/distances"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # ensure output dir exists

index = faiss.read_index(FAISS_INDEX_PATH)
expected_dim = index.d  # Get expected dimension from FAISS index
print(
    f"FAISS index loaded. Contains {index.ntotal} vectors of dimension {expected_dim}."
)
num_embeddings = index.ntotal
N = num_embeddings  # Use N for number of embeddings

all_files = glob(os.path.join(DATA_DIR, "*.mid"))
all_files.sort()
track_files = glob(os.path.join(TRACK_DIR, "*.mid"))
track_files.sort()

if len(all_files) != num_embeddings:
    print(
        f"Warning: Number of files ({len(all_files)}) does not match number of embeddings ({num_embeddings}). Check data consistency."
    )
    all_files = all_files[:num_embeddings]


all_tracks = [basename(t) for t in track_files]
print(f"found {len(all_tracks)} tracks:{all_tracks[:3]}")

# --- Map files to tracks (Needs to be done before iteration) ---
print("Mapping files to tracks...")
track_mapping = defaultdict(list)
file_to_track = {}
for i, f in enumerate(all_files):
    fname = basename(f)
    track_name = fname.split("_")[0]
    if track_name in all_tracks:  # only consider files belonging to known tracks
        track_mapping[track_name].append(i)
        file_to_track[i] = track_name
    # else:
    # Handle files not belonging to any identified track if necessary
    # print(f"Warning: File {fname} does not seem to belong to any known track.")
    # pass
print(f"Mapped {len(file_to_track)} files to {len(track_mapping)} tracks.")

# --- Initialize Statistics ---
print("Initializing statistics...")
global_min_dist = float("inf")
global_max_dist = float("-inf")
global_min_pair_idx = (None, None)
global_max_pair_idx = (None, None)
global_sum_dist = 0.0
global_sum_sq_dist = 0.0
global_count = 0
global_vector_sum = np.zeros(
    expected_dim, dtype=np.float64
)  # Use float64 for sum precision

# Use defaultdict for track stats, initialized when a track pair is first seen
track_accumulators = defaultdict(
    lambda: {
        "min_dist": float("inf"),
        "max_dist": float("-inf"),
        "min_pair_idx": (None, None),
        "max_pair_idx": (None, None),
        "sum_dist": 0.0,
        "sum_sq_dist": 0.0,
        "count": 0,
        "vector_sum": np.zeros(expected_dim, dtype=np.float64),
    }
)

intra_track_sum_dist = 0.0
intra_track_count = 0
inter_track_sum_dist = 0.0
inter_track_count = 0

# Temp storage for histogram data

# --- Histogram Setup (Manual Binning) ---
# NOTE: Adjust hist_min_range and hist_max_range based on expected distances!
num_bins = 50
hist_min_range = 0.00
hist_max_range = 10.0
bin_edges = np.linspace(hist_min_range, hist_max_range, num_bins + 1)
hist_counts = np.zeros(num_bins, dtype=np.int64)
hist_underflow_count = 0
hist_overflow_count = 0
# ----------------------------------------

# --- Iterative Calculation ---
print(f"Calculating statistics iteratively across {N} embeddings...")
total_pairs = N * (N - 1) // 2
processed_pairs = 0

with Progress() as progress:
    task = progress.add_task("[cyan]Processing pairs...", total=total_pairs)

    for i in range(N):
        try:
            emb_i = index.reconstruct(i).astype(
                np.float64
            )  # Fetch embedding i, use float64
        except Exception as e:
            print(f"Error reconstructing embedding {i}: {e}. Skipping.")
            continue

        track_i = file_to_track.get(i)

        for j in range(i + 1, N):
            try:
                emb_j = index.reconstruct(j).astype(np.float64)  # Fetch embedding j
            except Exception as e:
                print(
                    f"Error reconstructing embedding {j}: {e}. Skipping pair ({i}, {j})."
                )
                continue

            track_j = file_to_track.get(j)

            # Calculate distance and vector
            diff_vector = emb_i - emb_j
            # dist = np.linalg.norm(diff_vector) # Can be slower
            dist_sq = np.dot(diff_vector, diff_vector)
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0

            # --- Update Global Stats ---
            global_sum_dist += dist
            global_sum_sq_dist += dist_sq  # Sum of squares uses squared distance
            global_count += 1
            global_vector_sum += diff_vector  # Sum vectors i->j

            if dist < global_min_dist:
                global_min_dist = dist
                global_min_pair_idx = (i, j)

            if dist > global_max_dist:
                global_max_dist = dist
                global_max_pair_idx = (i, j)

            # --- Update Histogram Data (Manual Binning) ---
            if dist < hist_min_range:
                hist_underflow_count += 1
            elif dist >= hist_max_range:
                hist_overflow_count += 1
            else:
                # Find the correct bin index
                bin_index = np.searchsorted(bin_edges, dist, side="right") - 1
                # Ensure index is within bounds (should be due to checks above, but belt-and-suspenders)
                if 0 <= bin_index < num_bins:
                    hist_counts[bin_index] += 1
                # else: # This case should ideally not happen with the checks above
                #     print(f"Warning: Distance {dist} resulted in unexpected bin index {bin_index}")

            # --- Update Track/Inter/Intra Stats ---
            if track_i is not None and track_j is not None:
                if track_i == track_j:  # Intra-track
                    track = track_i
                    stats = track_accumulators[
                        track
                    ]  # Get or initialize accumulator for this track
                    stats["sum_dist"] += dist
                    stats["sum_sq_dist"] += dist_sq
                    stats["count"] += 1
                    stats["vector_sum"] += diff_vector

                    if dist < stats["min_dist"]:
                        stats["min_dist"] = dist
                        stats["min_pair_idx"] = (i, j)
                    if dist > stats["max_dist"]:
                        stats["max_dist"] = dist
                        stats["max_pair_idx"] = (i, j)

                    intra_track_sum_dist += dist
                    intra_track_count += 1
                else:  # Inter-track
                    inter_track_sum_dist += dist
                    inter_track_count += 1

            progress.update(task, advance=1)

# --- Finalize Statistics ---
print("--- Global Statistics ---")

if global_count > 0:
    global_avg_dist = global_sum_dist / global_count
    # Var = E[X^2] - (E[X])^2
    global_variance = (global_sum_sq_dist / global_count) - (global_avg_dist**2)
    # Handle potential floating point inaccuracies giving small negative variance
    global_std_dev = math.sqrt(max(0, global_variance))
    global_avg_vector = global_vector_sum / global_count
    global_avg_vector_magnitude = np.linalg.norm(global_avg_vector)

    min_file1 = (
        basename(all_files[global_min_pair_idx[0]])
        if global_min_pair_idx[0] is not None
        else "N/A"
    )
    min_file2 = (
        basename(all_files[global_min_pair_idx[1]])
        if global_min_pair_idx[1] is not None
        else "N/A"
    )
    max_file1 = (
        basename(all_files[global_max_pair_idx[0]])
        if global_max_pair_idx[0] is not None
        else "N/A"
    )
    max_file2 = (
        basename(all_files[global_max_pair_idx[1]])
        if global_max_pair_idx[1] is not None
        else "N/A"
    )

    print(f"Smallest Distance: {global_min_dist:.4f}")
    print(f"  Files: {min_file1}, {min_file2}")
    print(f"Largest Distance: {global_max_dist:.4f}")
    print(f"  Files: {max_file1}, {max_file2}")
    print(f"Average Distance: {global_avg_dist:.4f}")
    print("Median Distance: N/A (Requires storing all distances)")
    print(f"Standard Deviation of Distances: {global_std_dev:.4f}")
    print(f"Average Distance Vector Magnitude: {global_avg_vector_magnitude:.4f}")
else:
    print("No pairs processed, cannot calculate global statistics.")
    global_avg_dist = 0  # Define for plot


print("--- Track-Specific Statistics ---")
final_track_stats = {}
processed_tracks = set()

# Calculate stats for tracks that had pairs
for track, acc in track_accumulators.items():
    processed_tracks.add(track)
    count = acc["count"]
    stats = {}
    if count > 0:
        # Explicitly use numeric fields for calculation
        sum_dist = acc["sum_dist"]
        sum_sq_dist = acc["sum_sq_dist"]
        avg = sum_dist / count
        variance = (sum_sq_dist / count) - (avg**2)
        std_dev = math.sqrt(max(0, variance))
        avg_vec = acc["vector_sum"] / count  # vector_sum is ndarray, count is int
        avg_vec_mag = np.linalg.norm(avg_vec)
        min_dist = acc["min_dist"]  # Already a float
        max_dist = acc["max_dist"]  # Already a float
        min_f1 = (
            basename(all_files[acc["min_pair_idx"][0]])
            if acc["min_pair_idx"][0] is not None
            else "N/A"
        )
        min_f2 = (
            basename(all_files[acc["min_pair_idx"][1]])
            if acc["min_pair_idx"][1] is not None
            else "N/A"
        )
        max_f1 = (
            basename(all_files[acc["max_pair_idx"][0]])
            if acc["max_pair_idx"][0] is not None
            else "N/A"
        )
        max_f2 = (
            basename(all_files[acc["max_pair_idx"][1]])
            if acc["max_pair_idx"][1] is not None
            else "N/A"
        )

        stats = {
            "min_dist": min_dist,
            "min_files": (min_f1, min_f2),
            "max_dist": max_dist,
            "max_files": (max_f1, max_f2),
            "avg_dist": avg,
            "median_dist": "N/A",
            "std_dev_dist": std_dev,
            "avg_vector_mag": avg_vec_mag,
            "pair_count": count,
            "file_count": len(
                track_mapping.get(track, [])
            ),  # Get total files mapped to track
        }
        print(
            f"Track '{track}' ({stats['file_count']} files, {stats['pair_count']} pairs):"
        )
        print(
            f"  Smallest Distance: {stats['min_dist']:.4f} between {stats['min_files'][0]} and {stats['min_files'][1]}"
        )
        print(
            f"  Largest Distance: {stats['max_dist']:.4f} between {stats['max_files'][0]} and {stats['max_files'][1]}"
        )
        print(f"  Average Distance: {stats['avg_dist']:.4f}")
        print("  Median Distance: N/A")
        print(f"  Std Dev Distance: {stats['std_dev_dist']:.4f}")
        print(f"  Avg Vector Magnitude: {stats['avg_vector_mag']:.4f}")
    else:  # Should not happen if track is in accumulators, but handle anyway
        stats = {"pair_count": 0, "file_count": len(track_mapping.get(track, []))}
        print(
            f"Track '{track}' ({stats['file_count']} files, 0 pairs): No pairs processed."
        )

    final_track_stats[track] = stats


print("--- Inter vs Intra Track Statistics ---")
avg_intra_track_dist = (
    intra_track_sum_dist / intra_track_count if intra_track_count > 0 else 0
)
avg_inter_track_dist = (
    inter_track_sum_dist / inter_track_count if inter_track_count > 0 else 0
)

print(
    f"Average Intra-Track Distance: {avg_intra_track_dist:.4f} (calculated over {intra_track_count} pairs)"
)
print(
    f"Average Inter-Track Distance: {avg_inter_track_dist:.4f} (calculated over {inter_track_count} pairs)"
)


# --- Plotting ---
print("--- Generating Plot ---")
plot_filename = "distance_distribution.png"
plot_filepath = os.path.join(OUTPUT_DIR, plot_filename)

# Check if any distances were counted in the bins
total_counted_in_hist = hist_counts.sum()
if total_counted_in_hist > 0 or hist_underflow_count > 0 or hist_overflow_count > 0:
    plt.figure(figsize=(12, 7))

    # Use plt.stairs for plotting histogram from counts and edges
    plt.stairs(
        hist_counts,
        bin_edges,
        fill=True,
        color="skyblue",
        edgecolor="black",
        label="Frequency",
    )

    plot_title = "Distribution of Pairwise Euclidean Distances"
    plt.title(plot_title)
    plt.xlabel("Euclidean Distance Bins")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Add mean line (if calculated)
    if global_count > 0:
        plt.axvline(
            global_avg_dist,
            color="r",
            linestyle="dashed",
            linewidth=1,
            label=f"Mean: {global_avg_dist:.2f}",
        )

    # Add text about underflow/overflow
    info_text = f"Hist Range: [{hist_min_range:.2f}, {hist_max_range:.2f})\n"
    info_text += f"Counts in Range: {total_counted_in_hist}\n"
    if hist_underflow_count > 0:
        info_text += f"Counts < {hist_min_range:.2f}: {hist_underflow_count}\n"
    if hist_overflow_count > 0:
        info_text += f"Counts >= {hist_max_range:.2f}: {hist_overflow_count}"

    # Place text box in upper right corner
    plt.text(
        0.97,
        0.97,
        info_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.legend(loc="upper left")  # Adjust legend location if needed
    plt.tight_layout(
        rect=(0.0, 0.0, 0.95, 1.0)
    )  # Adjust layout to prevent text overlap
    try:
        plt.savefig(plot_filepath)
        print(f"Distance distribution plot saved to: {plot_filepath}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()  # close the plot to free memory
else:
    print("No distances counted in histogram range, skipping plot generation.")


# --- Generate Markdown Report ---
print("--- Generating Markdown Report ---")
report_filename = "embedding_distance_report.md"
report_filepath = os.path.join(OUTPUT_DIR, report_filename)

try:
    with open(report_filepath, "w") as f:
        f.write("# Embedding Distance Analysis Report\n\n")
        f.write(f"Analysis based on FAISS index: `{FAISS_INDEX_PATH}`\n")
        f.write(f"Data directory: `{DATA_DIR}`\n")
        f.write(f"Number of embeddings analyzed: {N}\n")
        f.write(f"Number of files considered: {len(all_files)}\n")
        f.write(f"Number of tracks identified: {len(track_mapping)}\n")
        f.write(f"Total pairs processed: {processed_pairs}\n\n")

        f.write("## Global Statistics\n\n")
        if global_count > 0:
            f.write(f"*   **Smallest Distance:** {global_min_dist:.4f}\n")
            f.write(f"    *   Files: `{min_file1}`, `{min_file2}`\n")
            f.write(f"*   **Largest Distance:** {global_max_dist:.4f}\n")
            f.write(f"    *   Files: `{max_file1}`, `{max_file2}`\n")
            f.write(f"*   **Average Distance:** {global_avg_dist:.4f}\n")
            f.write(f"*   **Median Distance:** N/A\n")
            f.write(f"*   **Standard Deviation:** {global_std_dev:.4f}\n")
            f.write(
                f"*   **Average Distance Vector Magnitude:** {global_avg_vector_magnitude:.4f}\n\n"
            )
        else:
            f.write("*   No pairs processed.\n\n")

        f.write("## Distance Distribution Plot\n\n")
        # Check if any distances were counted for the histogram
        if (
            total_counted_in_hist > 0
            or hist_underflow_count > 0
            or hist_overflow_count > 0
        ):
            f.write(
                f"The distribution of pairwise Euclidean distances (based on {global_count} pairs) is plotted below.\n"
            )
            f.write(
                f"Histogram generated using {num_bins} bins over the fixed range [{hist_min_range:.2f}, {hist_max_range:.2f}).\n\n"
            )
            if hist_underflow_count > 0:
                f.write(
                    f"*   Note: {hist_underflow_count} distances were below {hist_min_range:.2f}.\n"
                )
            if hist_overflow_count > 0:
                f.write(
                    f"*   Note: {hist_overflow_count} distances were at or above {hist_max_range:.2f}.\n"
                )
            f.write("\n")
            # Assuming the markdown file is in OUTPUT_DIR, the plot is in the same dir
            f.write(f"![Distance Distribution](./{plot_filename})\n\n")
        else:
            f.write("Plot skipped as no distances were counted for the histogram.\n\n")

        f.write("## Track-Specific Statistics\n\n")
        f.write("Statistics calculated for pairs of files within the same track.\n\n")
        # Sort tracks for consistent report order
        for track in sorted(final_track_stats.keys()):
            stats = final_track_stats[track]
            file_count = stats.get(
                "file_count", len(track_mapping.get(track, []))
            )  # Get file count
            pair_count = stats.get("pair_count", 0)

            f.write(f"### Track: `{track}` ({file_count} files)\n\n")
            if pair_count > 0:
                f.write(f"*   Pairs Processed: {pair_count}\n")
                f.write(
                    f"*   **Smallest Distance:** {stats['min_dist']:.4f} (`{stats['min_files'][0]}`, `{stats['min_files'][1]}`)\n"
                )
                f.write(
                    f"*   **Largest Distance:** {stats['max_dist']:.4f} (`{stats['max_files'][0]}`, `{stats['max_files'][1]}`)\n"
                )
                f.write(f"*   **Average Distance:** {stats['avg_dist']:.4f}\n")
                f.write(f"*   **Median Distance:** N/A\n")
                f.write(f"*   **Standard Deviation:** {stats['std_dev_dist']:.4f}\n")
                f.write(
                    f"*   **Average Vector Magnitude:** {stats['avg_vector_mag']:.4f}\n\n"
                )
            else:
                f.write(f"*   Pairs Processed: 0\n")
                f.write(
                    f"*   Skipped: Less than 2 files or no pairs processed for this track.\n\n"
                )

        f.write("## Inter vs. Intra-Track Comparison\n\n")
        f.write(
            f"*   **Average Intra-Track Distance:** {avg_intra_track_dist:.4f} (over {intra_track_count} pairs)\n"
        )
        f.write(
            f"*   **Average Inter-Track Distance:** {avg_inter_track_dist:.4f} (over {inter_track_count} pairs)\n\n"
        )

        if intra_track_count > 0 and inter_track_count > 0:
            if avg_inter_track_dist > avg_intra_track_dist:
                obs_str = "Observation: On average, files from different tracks are further apart than files within the same track, suggesting some track-level clustering in the embedding space.\n"
                f.write(obs_str)
            elif avg_intra_track_dist > avg_inter_track_dist:
                obs_str = "Observation: On average, files from different tracks are closer than files within the same track. This might indicate that track boundaries are not well-defined in the embedding space or that variation within tracks is high.\n"
                f.write(obs_str)
            else:
                obs_str = "Observation: Average intra-track and inter-track distances are very similar.\n"
                f.write(obs_str)
        else:
            f.write(
                "Observation: Cannot compare inter/intra distances as one or both categories have 0 pairs.\n"
            )

    print(f"Markdown report saved to: {report_filepath}")

except Exception as e:
    print(f"Error writing markdown report: {e}")


print("Analysis complete.")
