# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to cache models for stable diffusion."""
import argparse
import json
import logging
import re
import subprocess
import tempfile
import time

import requests
from huggingface_hub import login

from optimum.neuron import version as optimum_neuron_version


# Setup logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger()


def get_neuronx_cc_version():
    result = subprocess.run(["neuronx-cc", "--version"], capture_output=True, text=True)
    version_match = re.search(r"NeuronX Compiler version ([\d\.]+\+[a-f0-9]+)", result.stderr)
    if version_match:
        return version_match.group(1)
    else:
        raise ValueError("Version information not found in the output")


def get_aws_neuronx_tools_version():
    output = subprocess.check_output(["apt", "show", "aws-neuronx-tools"], text=True)
    version_match = re.search(r"Version: ([\d\.]+)", output)

    if version_match:
        # extract the version number and remove the last two characters (not tracked in optimum)
        return version_match.group(1)[:-2]
    else:
        raise ValueError("Version information not found in the output")


def compile_and_cache_model(
    hf_model_id, task, batch_size, height, width, num_images_per_prompt, auto_cast, auto_cast_type
):
    start = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Compile model with Optimum for specific configurations
        compile_command = [
            "optimum-cli",
            "export",
            "neuron",
            "-m",
            hf_model_id,
            "--task",
            task,
            "--batch_size",
            str(batch_size),
            "--height",
            str(height),
            "--width",
            str(width),
            "--num_images_per_prompt",
            str(num_images_per_prompt),
            "--auto_cast",
            auto_cast,
            "--auto_cast_type",
            auto_cast_type,
            temp_dir,
        ]
        logger.info(f"Running compile command: {' '.join(compile_command)}")
        try:
            subprocess.run(compile_command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to compile model: {e}")
            return

        # Synchronize compiled model to Hugging Face Hub
        cache_sync_command = ["optimum-cli", "neuron", "cache", "synchronize"]
        logger.info(f"Running cache synchronize command: {' '.join(cache_sync_command)}")

        try:
            subprocess.run(cache_sync_command, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to synchronize compiled model: {e}")
            return

    # Log time taken
    logger.info(f"Compiled and cached model {hf_model_id} w{time.time() - start:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile and cache a model to the Hugging Face Hub.")
    parser.add_argument("--hf_model_id", type=str, help="Hugging Face model ID to compile.")
    parser.add_argument("--task", type=str, help="Task for compilation.")
    parser.add_argument("--batch_size", type=int, help="Batch size for compilation.")
    parser.add_argument("--height", type=int, help="Image height for compilation.")
    parser.add_argument("--width", type=int, help="Image width for compilation.")
    parser.add_argument(
        "--num_images_per_prompt", type=int, help="Number of images generated per prompt for compilation."
    )
    parser.add_argument(
        "--auto_cast",
        type=str,
        choices=["none", "matmul", "all"],
        help="Cast operations from FP32 to lower precision to speed up the inference.",
    )
    parser.add_argument(
        "--auto_cast_type", type=str, choices=["bf16", "fp16", "tf32"], help="Auto cast type for compilation."
    )
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for authentication if not logged in.")
    parser.add_argument("--config_file", type=str, help="Path to a json config file with model configurations.")
    args = parser.parse_args()

    # Ensure either HF token is provided or user is already logged in
    if args.hf_token:
        logger.info(f"Logging in to Hugging Face Hub with {args.hf_token[:10]}...")
        login(args.hf_token)
    else:
        logger.info("Trying to use existing Hugging Face Hub login or environment variable HF_TOKEN")

    # check and get neuronx-cc version
    neuronx_cc_version = get_neuronx_cc_version()
    sdk_version = get_aws_neuronx_tools_version()
    logger.info(f"Compiler version: {neuronx_cc_version}")
    logger.info(f"Neuron SDK version: {sdk_version}")
    logger.info(f"Optimum Neuron version: {optimum_neuron_version.__version__}")
    logger.info(f"Compatible Optimum Neuron SDK version: {optimum_neuron_version.__sdk_version__} == {sdk_version}")
    assert (
        optimum_neuron_version.__sdk_version__ == sdk_version
    ), f"Optimum Neuron SDK version {optimum_neuron_version.__sdk_version__} is not compatible with installed Neuron SDK version {sdk_version}"

    # If a config file is provided, compile and cache all models in the file
    if args.config_file:
        logger.info(f"Compiling and caching models from config file: {args.config_file}")
        # check if config file starts with https://
        if args.config_file.startswith("https://"):
            response = requests.get(args.config_file)
            response.raise_for_status()
            config = response.json()
        else:
            with open(args.config_file, "r") as f:
                config = json.load(f)
        for model_id, configs in config.items():
            for model_config in configs:

                compile_and_cache_model(
                    hf_model_id=model_id,
                    task=model_config["task"],
                    batch_size=model_config["batch_size"],
                    height=model_config["height"],
                    width=model_config["width"],
                    num_images_per_prompt=model_config["num_images_per_prompt"],
                    auto_cast=model_config["auto_cast"],
                    auto_cast_type=model_config["auto_cast_type"],
                )
    elif (
        args.hf_model_id is None
        or args.task is None
        or args.batch_size is None
        or args.height is None
        or args.width is None
        or args.num_images_per_prompt is None
        or args.auto_cast is None
        or args.auto_cast_type is None
    ):
        raise ValueError(
            "You must provide a --hf_model_id, --task, --batch_size, --height, --width, --num_images_per_prompt, --auto_cast and --auto_cast_type to compile a model without a config file."
        )
    else:
        compile_and_cache_model(
            hf_model_id=args.hf_model_id,
            task=args.task,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            num_images_per_prompt=args.num_images_per_prompt,
            auto_cast=args.auto_cast,
            auto_cast_type=args.auto_cast_type,
        )
