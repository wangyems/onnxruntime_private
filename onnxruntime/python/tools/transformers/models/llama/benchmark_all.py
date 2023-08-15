import argparse
import datetime
import json
import logging
import os
import subprocess
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark_helper import setup_logger  # noqa: E402

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--batch-sizes",
        type=str,
        default="1 2",
    )

    parser.add_argument(
        "-s",
        "--sequence-lengths",
        type=str,
        default="8 16 32 64 128 256 512",
    )

    parser.add_argument(
        "-w",
        "--warmup-runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--hf-ort-model-path",
        type=str,
        help="Path to folder containing ONNX models for Optimum + ORT benchmarking",
    )

    parser.add_argument(
        "--ort-model-path",
        type=str,
        help="Path to ONNX model for ORT benchmarking",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name in Hugging Face",
    )

    parser.add_argument(
        "--model-size",
        type=str,
        required=True,
        choices=["7b", "13b", "70b"],
        help="Number of parameters in model",
    )

    parser.add_argument(
        "--precision",
        type=str,
        required=True,
        choices=["int8", "fp16", "fp32"],
        help="Precision to run model",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "cuda", "rocm"],
        help="Device to benchmark models",
    )

    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Print detailed logs",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Number of mins to attempt the benchmark before moving on",
    )

    args = parser.parse_args()

    log_folder_name = f"./{args.model_size}_{args.precision}"
    setattr(args, "log_folder", log_folder_name)  # noqa: B010
    os.makedirs(args.log_folder, exist_ok=True)

    # Convert timeout value to secs
    args.timeout *= 60

    return args


def process_log_file(device_id, log_file, base_results):
    entries = []
    batch_size, sequence_length, step = None, None, None
    latency_s, latency_ms, throughput, memory = None, None, None, None

    batch_pattern = "Batch Size: "
    sequence_pattern = "Sequence Length: "
    prompt_step_pattern = "to get past_key_values"
    per_token_step_pattern = "with past_key_values"
    latency_pattern = "Latency: "
    throughput_pattern = "Throughput: "
    memory_pattern = "peak="

    with open(log_file) as f:
        for input_line in f:
            line = input_line.replace("\n", "")

            if batch_pattern in line:
                batch_size = int(line[len(batch_pattern) :])
            elif sequence_pattern in line:
                sequence_length = int(line[len(sequence_pattern) :])
            elif prompt_step_pattern in line:
                step = "prompt"
            elif per_token_step_pattern in line:
                step = "per-token"
            elif latency_pattern in line:
                latency_s = float(line[len(latency_pattern) : line.rfind(" ")])
                latency_ms = latency_s * 1000
            elif throughput_pattern in line:
                throughput = float(line[len(throughput_pattern) : line.rfind(" ")])
            elif memory_pattern in line:
                if "CPU" in line:
                    # Example format for log entry:
                    # CPU memory usage: before=1000.0 MB, peak=2000.0 MB
                    memory = float(line[line.rfind("=") + 1 : line.rfind(" MB")]) / 1000
                else:
                    # Example format for log entry:
                    # GPU memory usage: before=[{'device_id': 0, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 69637.25}, {'device_id': 1, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 890.625}]  peak=[{'device_id': 0, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 73861.25}, {'device_id': 1, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 890.625}]
                    peak = line[line.find(memory_pattern) + len(memory_pattern) :].replace("'", '"')
                    usage = json.loads(peak)[device_id]["max_used_MB"]
                    memory = float(usage) / 1000

                # Append log entry to list of entries
                entry = base_results + [  # noqa: RUF005
                    batch_size,
                    sequence_length,
                    step,
                    latency_s,
                    latency_ms,
                    throughput,
                    memory,
                ]
                entries.append(entry)

    return entries


def save_results(results, filename):
    import pandas as pd

    df = pd.DataFrame(
        results,
        columns=[
            "Engine",
            "Precision",
            "Device",
            "Batch Size",
            "Sequence Length",
            "Step",
            "Latency (s)",
            "Latency (ms)",
            "Throughput (qps)",
            "Memory (GB)",
        ],
    )

    # Set column types
    df["Batch Size"] = df["Batch Size"].astype("int")
    df["Sequence Length"] = df["Sequence Length"].astype("int")
    df["Latency (s)"] = df["Latency (s)"].astype("float")
    df["Latency (ms)"] = df["Latency (ms)"].astype("float")
    df["Throughput (qps)"] = df["Throughput (qps)"].astype("float")
    df["Memory (GB)"] = df["Memory (GB)"].astype("float")

    df.to_csv(filename, index=False)
    logger.info(f"Results saved in {filename}!")


def benchmark(args, benchmark_cmd, engine):
    log_filename = f"{engine}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.log"
    log_path = os.path.join(args.log_folder, log_filename)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(benchmark_cmd, stdout=log_file, stderr=log_file)
        try:
            process.wait(args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    # Create entries for csv
    logger.info("Gathering data from log files...")
    base_results = [engine, args.precision, args.device]
    results = process_log_file(args.device_id, log_path, base_results)

    return results


def main():
    args = get_args()
    setup_logger(args.verbose)
    logger.info(args.__dict__)
    torch.backends.cudnn.benchmark = True

    all_results = []
    # Benchmark PyTorch without torch.compile
    benchmark_cmd = [
        "python3",
        "benchmark.py",
        "--benchmark-type",
        "hf-pt",
        "--model-name",
        args.model_name,
        "--model-size",
        args.model_size,
        "--precision",
        args.precision,
        "--batch-sizes",
        args.batch_sizes,
        "--sequence-lengths",
        args.sequence_lengths,
        "--device",
        args.device,
        "--device-id",
        str(args.device_id),
        "--warmup-runs",
        str(args.warmup_runs),
        "--num-runs",
        str(args.num_runs),
        "--log-folder",
        args.log_folder,
        "--auth",
    ]
    logger.info("Benchmark PyTorch without torch.compile")
    results = benchmark(args, benchmark_cmd, "pytorch")
    all_results.extend(results)

    # Benchmark PyTorch with torch.compile
    benchmark_cmd = [
        "python3",
        "benchmark.py",
        "--benchmark-type",
        "hf-pt2",
        "--model-name",
        args.model_name,
        "--model-size",
        args.model_size,
        "--precision",
        args.precision,
        "--batch-sizes",
        args.batch_sizes,
        "--sequence-lengths",
        args.sequence_lengths,
        "--device",
        args.device,
        "--device-id",
        str(args.device_id),
        "--warmup-runs",
        str(args.warmup_runs),
        "--num-runs",
        str(args.num_runs),
        "--log-folder",
        args.log_folder,
        "--auth",
    ]
    logger.info("Benchmark PyTorch with torch.compile")
    results = benchmark(args, benchmark_cmd, "pytorch-2")
    all_results.extend(results)

    # Benchmark Optimum + ONNX Runtime
    if args.hf_ort_model_path:
        benchmark_cmd = [
            "python3",
            "benchmark.py",
            "--benchmark-type",
            "hf-ort",
            "--hf-ort-model-path",
            args.hf_ort_model_path,
            "--model-name",
            args.model_name,
            "--model-size",
            args.model_size,
            "--precision",
            args.precision,
            "--batch-sizes",
            args.batch_sizes,
            "--sequence-lengths",
            args.sequence_lengths,
            "--device",
            args.device,
            "--device-id",
            str(args.device_id),
            "--warmup-runs",
            str(args.warmup_runs),
            "--num-runs",
            str(args.num_runs),
            "--log-folder",
            args.log_folder,
            "--auth",
        ]
        logger.info("Benchmark Optimum + ONNX Runtime")
        results = benchmark(args, benchmark_cmd, "pytorch-ort")
        all_results.extend(results)

    # Benchmark ONNX Runtime
    if args.ort_model_path:
        benchmark_cmd = [
            "python3",
            "benchmark.py",
            "--benchmark-type",
            "ort",
            "--ort-model-path",
            args.ort_model_path,
            "--model-name",
            args.model_name,
            "--model-size",
            args.model_size,
            "--precision",
            args.precision,
            "--batch-sizes",
            args.batch_sizes,
            "--sequence-lengths",
            args.sequence_lengths,
            "--device",
            args.device,
            "--device-id",
            str(args.device_id),
            "--warmup-runs",
            str(args.warmup_runs),
            "--num-runs",
            str(args.num_runs),
            "--log-folder",
            args.log_folder,
        ]
        logger.info("Benchmark ONNX Runtime")
        results = benchmark(args, benchmark_cmd, "onnxruntime")
        all_results.extend(results)

    csv_file = f"{args.model_size}_{args.precision}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.csv"
    save_results(all_results, os.path.join(args.log_folder, csv_file))


if __name__ == "__main__":
    main()
