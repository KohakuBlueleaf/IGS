import torch
import torch.nn as nn
import torch.nn.functional as F

from igs.gs_triton import TritonGaussianSplatting2D
from igs.gs_triton_chunked import TritonGaussianSplatting2DChunked
from igs.gs2d import TorchGaussianSplatting2DKernel, GaussianSplatting2DChunkedKernel
import torch
import torch.nn.functional as F
import triton
import itertools
import pandas as pd
from tqdm import tqdm, trange

# --- KERNELS TO BENCHMARK ---
# Add your kernel classes here. They must have the same `.apply` interface.

# This is the list the script will iterate over.
KERNELS_TO_BENCHMARK = [
    # TorchGaussianSplatting2DKernel,
    # GaussianSplatting2DChunkedKernel,
    TritonGaussianSplatting2D,
    TritonGaussianSplatting2DChunked,
]


# --- BENCHMARK CONFIGURATION ---

# Define the ranges for your problem dimensions
B_vals = [
    # 1,
    4,
    8,
    # 16,
]
N_vals = [
    4096,
    8192,
    16384,
]
H_vals = [
    # 64,
    128,
    256,
    # 512,
]
BLOCK_SIZES = [(-1, -1)]
# W is assumed to be the same as H for this benchmark
problem_sizes = list(itertools.product(B_vals, N_vals, H_vals, BLOCK_SIZES))


# --- BENCHMARKING FUNCTION ---


def create_inputs(B, N, H, W, requires_grad=False):
    """Helper function to create random input tensors."""
    return (
        torch.randn(B, 1, H, W, device="cuda", requires_grad=requires_grad),
        torch.randn(B, 1, H, W, device="cuda", requires_grad=requires_grad),
        torch.randn(B, N, 2, device="cuda", requires_grad=requires_grad),
        torch.randn(B, N, 1, 1, device="cuda", requires_grad=requires_grad),
        torch.randn(B, N, 1, 1, device="cuda", requires_grad=requires_grad),
        torch.randn(B, N, 1, 1, device="cuda", requires_grad=requires_grad),
        torch.rand(B, N, device="cuda", requires_grad=requires_grad),
        torch.rand(B, N, 3, device="cuda", requires_grad=requires_grad),
    )


def benchmark_pass(
    kernel_class, B, N, H, W, block_h, block_w, mode="fwd", warmup=4, rep=32
):
    """
    Runs a benchmark for a single configuration.

    Args:
        kernel_class: The torch.autograd.Function to benchmark.
        B, N, H, W: Problem dimensions.
        block_h, block_w: Block sizes for the kernel.
        mode: 'fwd' for forward pass only, 'fwd_bwd' for forward + backward.
        warmup: Number of warmup iterations.
        rep: Number of timed repetitions.

    Returns:
        Median execution time in milliseconds.
    """
    requires_grad = mode == "fwd_bwd"
    inputs = create_inputs(B, N, H, W, requires_grad=requires_grad)

    # Set block sizes
    if block_h > 0:
        kernel_class.BLOCK_SIZE_H = block_h
        kernel_class.BWD_BLOCK_SIZE_H = block_h
    if block_w > 0:
        kernel_class.BLOCK_SIZE_W = block_w
        kernel_class.BWD_BLOCK_SIZE_W = block_w

    def fwd_fn():
        # We clear the cache to prevent interference between different block sizes
        torch.cuda.empty_cache()
        return kernel_class.apply(*inputs)

    # --- Warmup Phase ---
    for _ in range(warmup):
        output = fwd_fn()
        if mode == "fwd_bwd":
            # Use a dummy gradient to trigger backward pass
            dummy_grad = torch.ones_like(output)
            output.backward(gradient=dummy_grad, retain_graph=False)
            # Reset grads
            for tensor in inputs:
                if tensor.grad is not None:
                    tensor.grad = None

    torch.cuda.synchronize()

    # --- Timed Repetitions ---
    if mode == "fwd":
        # Custom timing loop for forward + backward
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(rep):
            start_event.record()
            output = fwd_fn()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
            # Reset grads for the next iteration
            for tensor in inputs:
                if tensor.grad is not None:
                    tensor.grad = None
        return torch.tensor(times).median().item()

    elif mode == "fwd_bwd":
        # Custom timing loop for forward + backward
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(rep):
            start_event.record()
            output = fwd_fn()
            # Note: Using .mean().backward() as in your example
            output.mean().backward()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
            # Reset grads for the next iteration
            for tensor in inputs:
                if tensor.grad is not None:
                    tensor.grad = None
        return torch.tensor(times).median().item()

    else:
        print("Invalid mode:", mode)
        return float("inf")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    benchmark_results = []

    print("Starting benchmark...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Triton Version: {triton.__version__}")
    print("-" * 60)

    # Main loop over all configurations
    for B, N, H, (block_h, block_w) in (glob_pbar := tqdm(problem_sizes)):
        W = H  # Assuming square images
        glob_pbar.set_description(
            f"B={B}, N={N}, H={H}, W={W}, block={(block_h, block_w)}"
        )

        for kernel_class in (kernel_pbar := tqdm(KERNELS_TO_BENCHMARK, leave=False)):
            kernel_name = kernel_class.__name__
            kernel_pbar.set_description(kernel_name)

            # --- Autotuning for the best block configuration ---
            best_fwd_ms = float("inf")
            best_bwd_ms = float("inf")
            best_fwd_config = None
            best_bwd_config = None

            # Benchmark forward pass
            fwd_ms = benchmark_pass(
                kernel_class, B, N, H, W, block_h, block_w, mode="fwd"
            )
            bwd_ms = benchmark_pass(
                kernel_class, B, N, H, W, block_h, block_w, mode="fwd_bwd"
            )
            if fwd_ms < best_fwd_ms:
                best_fwd_ms = fwd_ms
                best_fwd_config = (block_h, block_w)
            if bwd_ms < best_bwd_ms:
                best_bwd_ms = bwd_ms
                best_bwd_config = (block_h, block_w)

            # --- Calculate metrics for the best configurations ---
            # Total operations for the "gaussians * pixels" metric
            gaussians_pixels = B * N * H * W

            # Forward pass metrics
            fwd_time_s = best_fwd_ms / 1000.0
            fwd_gpps = (gaussians_pixels / fwd_time_s) / 1e9  # Giga
            fwd_gps = (B * N / fwd_time_s) / 1e6  # Mega

            # Forward + Backward pass metrics
            bwd_time_s = best_bwd_ms / 1000.0
            bwd_gpps = (gaussians_pixels / bwd_time_s) / 1e9
            bwd_gps = (B * N / bwd_time_s) / 1e6

            # Store results
            benchmark_results.append(
                {
                    "Kernel": kernel_name,
                    "B": B,
                    "N": N,
                    "H": H,
                    "W": W,
                    "block": (block_h, block_w),
                    "Pass": "Forward",
                    "ms": f"{best_fwd_ms:.4f}",
                    "G-pixels/s": f"{fwd_gpps:.2f}",
                    "M-Gaussians/s": f"{fwd_gps:.2f}",
                }
            )
            benchmark_results.append(
                {
                    "Kernel": kernel_name,
                    "B": B,
                    "N": N,
                    "H": H,
                    "W": W,
                    "block": (block_h, block_w),
                    "Pass": "Fwd+Bwd",
                    "ms": f"{best_bwd_ms:.4f}",
                    "G-pixels/s": f"{bwd_gpps:.2f}",
                    "M-Gaussians/s": f"{bwd_gps:.2f}",
                }
            )

    # --- Print final results table ---
    print("\n" + "=" * 80)
    print("Benchmark Results Summary")
    print("=" * 80)

    df = pd.DataFrame(benchmark_results)

    # Reorder columns for better readability
    column_order = [
        "Kernel",
        "B",
        "N",
        "H",
        "block",
        "Pass",
        "ms",
        "G-pixels/s",
        "M-Gaussians/s",
    ]
    # Filter out W if it's always the same as H
    if all(df["W"] == df["H"]):
        df = df.drop(columns="W")
    else:
        column_order.insert(4, "W")

    print(df.to_string(index=False, columns=column_order))
    print("-" * 80)
