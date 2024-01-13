#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    input_data = pd.read_csv(artifact_local_path)
    mask_price_range = input_data["price"].between(args.min_price, args.max_price)
    logger.info(f"we have in total {len(mask_price_range)} observations.")

    input_data = input_data[mask_price_range].copy()

    logger.info(f"Convert last_review to datetime")
    input_data["last_review"] = pd.to_datetime(input_data["last_review"])

    logger.info(f"Drop rows in the dataset that are not in the proper geolocation.")
    idx = input_data["longitude"].between(-74.25, -73.50) & input_data[
        "latitude"
    ].between(40.5, 41.2)
    input_data = input_data[idx].copy()

    input_data.to_csv(f"{args.output_artifact}", index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(f"{args.output_artifact}")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact", type=str, help="Name of input artifact", required=True
    )

    parser.add_argument(
        "--output_artifact", type=str, help="Name of output artifact", required=True
    )

    parser.add_argument(
        "--output_type", type=str, help="Output artifact type", required=True
    )

    parser.add_argument(
        "--output_description", type=str, help="Output description", required=True
    )

    parser.add_argument(
        "--min_price", type=float, help="Floor for price model", required=True
    )

    parser.add_argument(
        "--max_price", type=float, help="Cap for price model", required=True
    )

    args = parser.parse_args()

    go(args)
