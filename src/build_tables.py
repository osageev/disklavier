import os
import time
import glob
from omegaconf import OmegaConf
from argparse import ArgumentParser

from utils import console
from utils.tables import build_neighbor_table, pitch_histograms, specdiff, classifier

SUPPORTED_REPS = ["pitch-histogram", "specdiff", "clf-4note", "clf-speed", "clf-tpose"]


def main(config) -> None:
    start_time = time.time()
    # load files
    segmented_files = glob.glob(os.path.join(config.out_dir, "segmented", "*.mid"))
    segmented_files.sort()
    console.log(f"loaded segmented files", segmented_files[:5])
    augmented_files = glob.glob(os.path.join(config.out_dir, "augmented", "*.mid"))
    augmented_files.sort()
    console.log(f"loaded augmented files", augmented_files[:5])

    # build neighbor table
    if config.find_neighbors:
        neighbors_path = os.path.join(config.out_dir, "neighbors.h5")
        if os.path.exists(neighbors_path):
            console.log(f"neighbor table already exists, delete or remove it")
        else:
            console.log(
                f"building neighbor table for {len(segmented_files)} segmented files"
            )
            built_n = build_neighbor_table(
                segmented_files, neighbors_path, config.dataset_name
            )
            if built_n:
                console.log(
                    f"built neighbor table in {time.time() - start_time:.03f} s"
                )
            else:
                console.log(f"failed to build neighbor table")
                exit(1)

    # now the fun part, calculating all the representations
    for rep in config.representations:
        table_path = os.path.join(config.out_dir, f"{rep}.h5")
        if os.path.exists(table_path):
            console.log(f"skipping {rep} because it already exists")
            continue

        match rep:
            case "pitch-histogram":
                resp = pitch_histograms(augmented_files, table_path)
                console.log(
                    f"pitch histogram table built [{resp}] in {time.time() - start_time:.03f} s at '{table_path}'"
                )
            case "specdiff":
                specdiff(augmented_files, table_path, device_name=config.device_name)
            case "clf-4note" | "clf-speed" | "clf-tpose":
                if not os.path.exists(os.path.join(config.out_dir, "specdiff.h5")):
                    console.log("need to generate specdiff table first")
                    specdiff(
                        augmented_files,
                        os.path.join(config.out_dir, "specdiff.h5"),
                        device_name=config.device_name,
                    )
                clf_path = os.path.join(config.model_dir, f"{rep}.pth")
                if not os.path.exists(clf_path):
                    console.log(f"classifier model not found at '{clf_path}'")
                    continue

                console.log(f"building {rep} table")
                classifier(
                    augmented_files,
                    table_path,
                    clf_path,
                    device_name=config.device_name,
                )
            case _:
                console.log(f"skipping {rep} because it is not supported")
                console.log(f"supported representations are: {SUPPORTED_REPS}")
                continue

    # generate FAISS indices
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset builder arguments")
    parser.add_argument(
        "-c", "--config", type=str, default=None, help="path to config file"
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    console.log(config)
    main(config)
