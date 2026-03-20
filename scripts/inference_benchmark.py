"""
Inference latency and throughput benchmark for ModernFinBERT.
Measures single-sample latency (p50/p95/p99) and batch throughput on CPU and GPU.
Outputs results as a formatted table and saves to results/inference_benchmark.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "inference_benchmark.json"

MODEL_ID = "neoyipeng/ModernFinBERT-base"

# Realistic financial text samples of varying lengths
SAMPLE_TEXTS = [
    "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007.",
    "Revenue declined 5% year-over-year.",
    "The company reported quarterly earnings that met analyst expectations, with strong performance "
    "in the consumer electronics division offsetting weakness in the enterprise segment.",
    "Net sales increased by 4% to EUR 1.2 billion.",
    "According to the CEO, the acquisition will strengthen the company's position in the Nordic "
    "market and is expected to generate annual synergies of approximately EUR 15 million within "
    "three years of closing.",
    "Finnish IT company Affecto Oyj HEL : AFE1V said today its net loss narrowed to EUR 0.5 "
    "million in the first quarter of 2010 from EUR 2.4 million in the corresponding period a "
    "year earlier.",
    "Shares dropped 8% in after-hours trading following the guidance cut.",
    "The Board of Directors proposes a dividend of EUR 0.12 per share for the financial year 2009.",
    "TeliaSonera reported strong growth in mobile services across all Nordic and Baltic markets, "
    "driven by increased data consumption and successful upselling of premium plans, though "
    "fixed-line revenue continued its structural decline amid ongoing fiber migration.",
    "Profit before taxes was EUR 4.0 mn, down from EUR 4.9 mn.",
]


def load_model(device):
    """Load ModernFinBERT model and tokenizer onto the specified device."""
    print(f"Loading {MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model = model.to(device).eval()
    return model, tokenizer


def measure_single_latency(model, tokenizer, device, warmup=10, iterations=100):
    """Measure single-sample inference latency with warmup.

    Returns dict with p50, p95, p99 latency in milliseconds.
    """
    text = SAMPLE_TEXTS[0]

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs).logits

    # Synchronize GPU before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs).logits

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    return {
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "mean_ms": round(float(np.mean(latencies)), 2),
        "std_ms": round(float(np.std(latencies)), 2),
    }


def measure_batch_throughput(model, tokenizer, device, batch_sizes=(1, 8, 32, 64),
                              warmup=5, iterations=20):
    """Measure throughput at different batch sizes.

    Returns dict mapping batch_size -> samples/second.
    """
    results = {}

    for bs in batch_sizes:
        # Build a batch by cycling through sample texts
        batch_texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(bs)]

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs).logits

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _ = model(**inputs).logits

                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)

        avg_time = np.mean(times)
        throughput = bs / avg_time

        results[bs] = {
            "batch_size": bs,
            "avg_time_ms": round(float(avg_time * 1000), 2),
            "throughput_samples_per_sec": round(float(throughput), 1),
            "per_sample_ms": round(float(avg_time * 1000 / bs), 2),
        }

    return results


def print_results(device_name, latency, throughput):
    """Print a formatted results table."""
    print(f"\n{'=' * 65}")
    print(f"  {device_name} — Single-Sample Latency ({MODEL_ID})")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<20} {'Value':>10}")
    print(f"  {'-' * 35}")
    print(f"  {'p50 latency':<20} {latency['p50_ms']:>8.2f} ms")
    print(f"  {'p95 latency':<20} {latency['p95_ms']:>8.2f} ms")
    print(f"  {'p99 latency':<20} {latency['p99_ms']:>8.2f} ms")
    print(f"  {'Mean latency':<20} {latency['mean_ms']:>8.2f} ms")
    print(f"  {'Std dev':<20} {latency['std_ms']:>8.2f} ms")

    print(f"\n{'=' * 65}")
    print(f"  {device_name} — Batch Throughput")
    print(f"{'=' * 65}")
    print(f"  {'Batch Size':>10}  {'Avg Time (ms)':>14}  {'Per Sample (ms)':>16}  {'Throughput':>12}")
    print(f"  {'-' * 60}")
    for bs in sorted(throughput.keys()):
        r = throughput[bs]
        print(f"  {bs:>10}  {r['avg_time_ms']:>14.2f}  {r['per_sample_ms']:>16.2f}  "
              f"{r['throughput_samples_per_sec']:>9.1f} s/s")


def main():
    all_results = {"model": MODEL_ID, "sample_texts_count": len(SAMPLE_TEXTS)}

    # CPU benchmark
    device_cpu = torch.device("cpu")
    model_cpu, tokenizer_cpu = load_model(device_cpu)

    print("\nBenchmarking on CPU...")
    cpu_latency = measure_single_latency(model_cpu, tokenizer_cpu, device_cpu)
    cpu_throughput = measure_batch_throughput(model_cpu, tokenizer_cpu, device_cpu)
    print_results("CPU", cpu_latency, cpu_throughput)

    all_results["cpu"] = {"latency": cpu_latency, "throughput": cpu_throughput}

    # Free CPU model
    del model_cpu
    del tokenizer_cpu

    # GPU benchmark (if available)
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
        model_gpu, tokenizer_gpu = load_model(device_gpu)

        print("\nBenchmarking on GPU...")
        gpu_latency = measure_single_latency(model_gpu, tokenizer_gpu, device_gpu)
        gpu_throughput = measure_batch_throughput(model_gpu, tokenizer_gpu, device_gpu)
        print_results("GPU", gpu_latency, gpu_throughput)

        all_results["gpu"] = {"latency": gpu_latency, "throughput": gpu_throughput}
        all_results["gpu_name"] = torch.cuda.get_device_name(0)

        del model_gpu
        del tokenizer_gpu
    elif torch.backends.mps.is_available():
        device_mps = torch.device("mps")
        model_mps, tokenizer_mps = load_model(device_mps)

        print("\nBenchmarking on MPS (Apple Silicon)...")
        mps_latency = measure_single_latency(model_mps, tokenizer_mps, device_mps)
        mps_throughput = measure_batch_throughput(model_mps, tokenizer_mps, device_mps)
        print_results("MPS (Apple Silicon)", mps_latency, mps_throughput)

        all_results["mps"] = {"latency": mps_latency, "throughput": mps_throughput}

        del model_mps
        del tokenizer_mps
    else:
        print("\nNo GPU available — skipping GPU benchmark.")

    # Save results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
