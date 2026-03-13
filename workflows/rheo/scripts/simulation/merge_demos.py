# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Merge multiple HDF5 demo files into one (supports both recorded and annotated demos)

Usage:
    # Merge recorded demos
    python merge_demos.py \
        --input demo_1.hdf5 demo_2.hdf5 demo_3.hdf5 \
        --output merged_demos.hdf5

    # Inspect a demo file (check if annotated)
    python merge_demos.py --inspect demo_1_annotated.hdf5
"""

import argparse
import os

import h5py


def merge_hdf5_demos(input_files: list[str], output_file: str, verbose: bool = True):
    """
    Merge multiple HDF5 demo files

    Args:
        input_files: List of input HDF5 file paths
        output_file: Output merged HDF5 file path
        verbose: Whether to print detailed information
    """

    # Check if input files exist
    for input_file in input_files:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Remove output file if it exists
    if os.path.exists(output_file):
        if verbose:
            print(f"Warning: Output file exists, will be overwritten: {output_file}")
        os.remove(output_file)

    # Create output file
    with h5py.File(output_file, "w") as out_f:
        # Create 'data' group to match IsaacLab format
        data_group = out_f.create_group("data")

        # Initialize global attributes and metadata
        total_episodes = 0

        # Iterate through each input file
        for file_idx, input_file in enumerate(input_files):
            if verbose:
                print(f"\n{'=' * 60}")
                print(f"Processing file {file_idx + 1}/{len(input_files)}: {input_file}")
                print(f"{'=' * 60}")

            with h5py.File(input_file, "r") as in_f:
                # Check if demos are nested under 'data' key
                root = in_f
                if "data" in in_f.keys() and len(in_f.keys()) == 1:
                    if verbose:
                        print("  Demos are nested under 'data' key")
                    root = in_f["data"]

                # Get all demo names (typically "demo_0", "demo_1", etc.)
                demo_keys = [key for key in root.keys() if key.startswith("demo_")]

                if verbose:
                    print(f"Found {len(demo_keys)} demos in this file")

                # Copy each demo
                for demo_key in demo_keys:
                    # Generate new demo name
                    new_demo_name = f"demo_{total_episodes}"

                    if verbose:
                        print(f"  Copying {demo_key} -> {new_demo_name}")

                    # Recursively copy entire demo group
                    def copy_group(src_group, dst_group):
                        """Recursively copy HDF5 group and all its contents"""
                        for key, item in src_group.items():
                            if isinstance(item, h5py.Group):
                                # Create subgroup and recursively copy
                                sub_group = dst_group.create_group(key)
                                copy_group(item, sub_group)
                                # Copy attributes
                                for attr_name, attr_value in item.attrs.items():
                                    sub_group.attrs[attr_name] = attr_value
                            elif isinstance(item, h5py.Dataset):
                                # Copy dataset
                                dst_group.create_dataset(
                                    key,
                                    data=item[...],
                                    compression=item.compression,
                                    compression_opts=item.compression_opts,
                                )
                                # Copy attributes
                                for attr_name, attr_value in item.attrs.items():
                                    dst_group[key].attrs[attr_name] = attr_value

                    # Create new demo group under 'data'
                    demo_group = data_group.create_group(new_demo_name)
                    copy_group(root[demo_key], demo_group)

                    # Copy demo group attributes
                    for attr_name, attr_value in root[demo_key].attrs.items():
                        demo_group.attrs[attr_name] = attr_value

                    total_episodes += 1

                # Copy attributes from first file
                if file_idx == 0:
                    # Copy root-level attributes
                    for attr_name, attr_value in in_f.attrs.items():
                        out_f.attrs[attr_name] = attr_value
                        if verbose:
                            print(f"  Copied root attribute: {attr_name}")

                    # Copy data group attributes (critical for IsaacLab!)
                    if "data" in in_f.keys():
                        for attr_name, attr_value in in_f["data"].attrs.items():
                            data_group.attrs[attr_name] = attr_value
                            if verbose:
                                print(f"  Copied data attribute: {attr_name}")

        # Update total demo count in data group
        data_group.attrs["total"] = total_episodes

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Successfully merged {len(input_files)} files")
            print(f"   Total episodes: {total_episodes}")
            print(f"   Output file: {output_file}")
            print(f"   File size: {os.path.getsize(output_file) / (1024**2):.2f} MB")
            print(f"{'=' * 60}")

    return total_episodes


def inspect_hdf5(file_path: str):
    """Inspect basic information of HDF5 file"""
    print(f"\nInspecting: {file_path}")
    print("-" * 60)

    with h5py.File(file_path, "r") as f:
        # Print top-level keys
        print(f"Top-level keys: {list(f.keys())}")

        # Print global attributes
        print("\nGlobal attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  {attr_name}: {attr_value}")

        # Check if data is nested under 'data' key
        root = f
        if "data" in f.keys() and len(f.keys()) == 1:
            print("\nData appears to be nested under 'data' key")
            root = f["data"]
            print(f"Keys under 'data': {list(root.keys())[:20]}")  # Show first 20

        # Count episodes (try both 'demo_' and 'episode_' prefixes)
        demo_keys = [key for key in root.keys() if key.startswith("demo_")]
        episode_keys = [key for key in root.keys() if key.startswith("episode_")]

        if demo_keys:
            print(f"\nNumber of demos: {len(demo_keys)}")
            print("\nDemo list:")
            annotated_count = 0
            for demo_key in sorted(demo_keys)[:10]:  # Show first 10
                demo = root[demo_key]
                success = demo.attrs.get("success", "Unknown")
                if "actions" in demo:
                    num_steps = len(demo["actions"])
                else:
                    num_steps = "Unknown"

                # Check if annotated (has subtask_term_signals)
                if (
                    "obs" in demo
                    and "datagen_info" in demo["obs"]
                    and "subtask_term_signals" in demo["obs/datagen_info"]
                ):
                    annotated_count += 1
                    annotation_status = "[OK] Annotated"
                else:
                    annotation_status = "[NO] Not annotated"

                print(f"  {demo_key}: {num_steps} steps, success={success}, {annotation_status}")

            if len(demo_keys) > 10:
                print(f"  ... and {len(demo_keys) - 10} more")

            # Check annotation completeness
            if annotated_count > 0:
                print("\nAnnotation Statistics:")
                print(f"  Annotated demos: {annotated_count}/{len(demo_keys)}")

                # Show subtask signals from first annotated demo
                for demo_key in sorted(demo_keys):
                    demo = root[demo_key]
                    if (
                        "obs" in demo
                        and "datagen_info" in demo["obs"]
                        and "subtask_term_signals" in demo["obs/datagen_info"]
                    ):
                        signals = demo["obs/datagen_info/subtask_term_signals"]
                        print(f"\n  Subtask signals in {demo_key}:")
                        for signal_name in signals.keys():
                            signal_data = signals[signal_name]
                            has_true = any(signal_data[()])
                            status = "[OK]" if has_true else "[!]"
                            print(f"    {status} {signal_name}: shape={signal_data.shape}, has_completion={has_true}")
                        break  # Only show first annotated demo

        elif episode_keys:
            print(f"\nNumber of episodes: {len(episode_keys)}")
            print("\nEpisode list:")
            for ep_key in sorted(episode_keys)[:10]:  # Show first 10
                episode = root[ep_key]
                if "actions" in episode:
                    num_steps = len(episode["actions"])
                else:
                    num_steps = "Unknown"
                print(f"  {ep_key}: {num_steps} steps")
            if len(episode_keys) > 10:
                print(f"  ... and {len(episode_keys) - 10} more")
        else:
            print("\nNumber of demos/episodes: 0")
            print("(No keys starting with 'demo_' or 'episode_' found)")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 demo files into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge 3 files
  python merge_demos.py --input demo_1.hdf5 demo_2.hdf5 demo_3.hdf5 --output merged.hdf5

  # Merge all demo files in current directory
  python merge_demos.py --input demo_*.hdf5 --output all_demos.hdf5

  # Inspect file information
  python merge_demos.py --inspect demo_1.hdf5
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        help="Input HDF5 files to merge",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output merged HDF5 file",
    )

    parser.add_argument(
        "--inspect",
        type=str,
        help="Inspect an HDF5 file without merging",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Inspect mode
    if args.inspect:
        inspect_hdf5(args.inspect)
        return None

    # Check parameters
    if not args.input or not args.output:
        parser.error("--input and --output are required (unless using --inspect)")

    # Expand wildcards (if using shell wildcards)
    input_files = []
    for pattern in args.input:
        if "*" in pattern or "?" in pattern:
            # Use glob to expand
            from glob import glob

            matched_files = glob(pattern)
            if not matched_files:
                print(f"Warning: Pattern '{pattern}' matched no files")
            input_files.extend(matched_files)
        else:
            input_files.append(pattern)

    if not input_files:
        parser.error("No input files found")

    # Remove duplicates and sort
    input_files = sorted(set(input_files))

    if not args.quiet:
        print(f"Found {len(input_files)} input files:")
        for f in input_files:
            print(f"  - {f}")

    # Merge files
    try:
        total_episodes = merge_hdf5_demos(
            input_files,
            args.output,
            verbose=not args.quiet,
        )

        if not args.quiet:
            print(f"\n{total_episodes} episodes merged successfully!")
            print("\nYou can now use the merged file for annotation:")
            print(f"  python annotate_demos.py --input_file {args.output} ...")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
