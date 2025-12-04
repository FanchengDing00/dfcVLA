#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from openpi.shared import download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_suffix",
        required=True,
        help='Checkpoint suffix after "gs://openpi-assets/checkpoints/", e.g. "pi05_libero" or "pi0_base".',
    )
    parser.add_argument(
        "--download_to_dir",
        required=True,
        help="Local directory for OPENPI_DATA_HOME (cache root). Checkpoints will be stored under <download-to-dir>/openpi-assets/checkpoints/",
    )
    args = parser.parse_args()

    # Set cache root
    os.environ["OPENPI_DATA_HOME"] = args.download_to_dir

    # Build full GCS path
    ckpt_gcs_path = f"gs://openpi-assets/checkpoints/{args.checkpoint_suffix}"

    # Download
    ckpt_dir = download.maybe_download(ckpt_gcs_path)
    print(f"Checkpoint downloaded to: {ckpt_dir}")

if __name__ == "__main__":
    main()
