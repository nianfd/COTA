# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Data preparation script for VisA dataset, modified from the PromptAD repository.

import argparse
import shutil
import csv
from PIL import Image
import numpy as np
from pathlib import Path
from typing import NamedTuple


class Config(NamedTuple):
    """Configuration settings"""

    split_type: str
    data_folder: Path
    save_folder: Path
    split_file: Path


def setup_arguments() -> Config:
    """Parses and returns command-line arguments."""
    parser = argparse.ArgumentParser(description="Data preparation")
    parser.add_argument(
        "--split-type",
        default="1cls",
        type=str,
        help="1cls, 2cls_highshot, 2cls_fewshot",
    )
    parser.add_argument(
        "--data-folder",
        default="./anomaly_detection/VisA_20220922",
        type=Path,
        help="the path to downloaded VisA dataset",
    )
    parser.add_argument(
        "--save-folder",
        default="./anomaly_detection/VisA_20220922/VisA_pytorch/",
        type=Path,
        help="the target path to save the reorganized VisA dataset",
    )
    parser.add_argument(
        "--split-file",
        default="./datasets/VisA_20220922/split_csv/1cls.csv",
        type=Path,
        help="the csv file to split downloaded VisA dataset",
    )

    args = parser.parse_args()

    return Config(
        split_type=args.split_type,
        data_folder=args.data_folder,
        save_folder=args.save_folder,
        split_file=args.split_file,
    )


def binarize_and_save_mask(src_path: Path, dst_path: Path) -> None:
    """Loads a mask, binarizes it (0 or 255), and saves it."""
    try:
        with Image.open(src_path) as mask:
            # binarize mask
            mask_array = np.array(mask)
            mask_array[mask_array != 0] = 255
            mask_image = Image.fromarray(mask_array)

            # Ensure destination directory exists
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            mask_image.save(dst_path)
    except FileNotFoundError:
        print(f"Warning: Mask file not found at {src_path}")
    except Exception as e:
        print(f"Error processing mask {src_path}: {e}")


def main():
    """Main data preparation script."""
    config = setup_arguments()

    # The final save folder depends on the split type
    save_folder = config.save_folder / config.split_type
    print(f"Starting data preparation for split: {config.split_type}")
    print(f"Reading data from: {config.data_folder}")
    print(f"Saving data to:   {save_folder}")
    print(f"Using split file: {config.split_file}")

    try:
        with open(config.split_file, "r", encoding="utf-8") as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            print(f"CSV Headers: {header}")

            for row in csvreader:
                category, data_split, label_str, image_path, mask_path = row

                # Map labels
                label = "good" if label_str == "normal" else "bad"

                # Use pathlib for clean path handling
                image_name = Path(image_path).name
                mask_name = Path(mask_path).name

                img_src_path = config.data_folder / image_path
                msk_src_path = config.data_folder / mask_path

                # --- Image Copying (Common to all split types) ---
                img_dst_path = save_folder / category / data_split / label / image_name

                # Create destination directory and copy image
                img_dst_path.parent.mkdir(parents=True, exist_ok=True)
                if img_src_path.exists():
                    shutil.copyfile(img_src_path, img_dst_path)
                else:
                    print(f"Warning: Image file not found at {img_src_path}")
                    continue

                # Mask copying
                should_save_mask = False
                msk_dst_path = None

                if config.split_type == "1cls":
                    # For 1cls, only save masks for test/bad
                    if data_split == "test" and label == "bad":
                        msk_dst_path = (
                            save_folder / category / "ground_truth" / label / mask_name
                        )
                        should_save_mask = True
                else:
                    # For other types, save all 'bad' masks, sorted by split
                    if label == "bad":
                        msk_dst_path = (
                            save_folder
                            / category
                            / "ground_truth"
                            / data_split
                            / label
                            / mask_name
                        )
                        should_save_mask = True

                # Process and save the mask if needed
                if should_save_mask:
                    if not mask_path or not msk_src_path.exists():
                        print(
                            f"Warning: Missing mask for bad sample: {image_path} (Expected at {msk_src_path})"
                        )
                        continue
                    binarize_and_save_mask(msk_src_path, msk_dst_path)

    except FileNotFoundError:
        print(f"Error: Split file not found at {config.split_file}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
