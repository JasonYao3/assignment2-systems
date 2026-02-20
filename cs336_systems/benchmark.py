import time
import torch
import logging
from cs336_basics.model import BasicsTransformerLM
import timeit
from contextlib import nullcontext

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_forward(model, input_data, num_steps, num_warmup_steps, use_mixed_precision=False):
    """
    Benchmarks the forward pass of the model.

    Args:
        model: The model to benchmark.
        input_data: The input data for the model.
        num_steps: The number of steps to measure.
        num_warmup_steps: The number of warmup steps.
    """
    if use_mixed_precision:
        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    # Warmup loop: run forward pass without tracking gradients or timing
    for _ in range(num_warmup_steps):
        with torch.no_grad():
            with ctx:
                output = model(input_data)
        torch.mps.synchronize()

    # Measurement loop
    time_taken_list = []
    for i in range(num_steps):
        torch.mps.synchronize()
        start = timeit.default_timer()
        with torch.no_grad():
            with ctx:
                output = model(input_data)

        end = timeit.default_timer()
        torch.mps.synchronize()
        time_taken = end - start
        time_taken_list.append(time_taken)

    mean_time = sum(time_taken_list) / len(time_taken_list)
    print(f"Forward Pass Benchmark:")
    print(f"  - Average Time per Step: {mean_time:.5f} seconds over {num_steps} steps")
    print(f"  - Throughput: {1 / mean_time:.2f} steps/second\n")


def benchmark_forward_backward(model, input_data, num_steps, num_warmup_steps, use_mixed_precision=False):
    """
    Benchmarks the forward and backward pass of the model.
    """
    if use_mixed_precision:
        ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    # Warmup loop: calculate gradients without timing
    for _ in range(num_warmup_steps):
        torch.mps.synchronize()
        model.zero_grad()
        with ctx:
            output = model(input_data)
        loss = output.sum()
        loss.backward()
        torch.mps.synchronize()

    # Measurement loop
    time_taken_list = []
    for i in range(num_steps):
        torch.mps.synchronize()
        start = timeit.default_timer()
        model.zero_grad()
        with ctx:
            output = model(input_data)
        loss = output.sum()
        loss.backward()
        end = timeit.default_timer()
        torch.mps.synchronize()
        time_taken = end - start
        time_taken_list.append(time_taken)

    mean_time = sum(time_taken_list) / len(time_taken_list)
    print(f"Forward + Backward Pass Benchmark:")
    print(f"  - Average Time per Step: {mean_time:.5f} seconds over {num_steps} steps")
    print(f"  - Throughput: {1 / mean_time:.2f} steps/second\n")


if __name__ == "__main__":
    # Define hyperparameters
    vocab_size = 1000
    batch_size = 16
    context_length = 5
    d_model = 128
    num_layers = 2
    num_heads = 2
    d_ff = 1
    rope_theta = 0.1

    print("Initializing model and data...")
    btlm = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    btlm = btlm.to("mps")
    input_data = torch.randint(vocab_size, (batch_size, context_length)).to("mps")

    print("Running FP32 benchmarks...\n")
    benchmark_forward(
        model=btlm,
        input_data=input_data,
        num_steps=10,
        num_warmup_steps=5,
        use_mixed_precision=False,
    )
    benchmark_forward_backward(
        model=btlm,
        input_data=input_data,
        num_steps=10,
        num_warmup_steps=5,
        use_mixed_precision=False,
    )

    print("Running BF16 Mixed Precision benchmarks...\n")
    benchmark_forward(
        model=btlm,
        input_data=input_data,
        num_steps=10,
        num_warmup_steps=5,
        use_mixed_precision=True,
    )
    benchmark_forward_backward(
        model=btlm,
        input_data=input_data,
        num_steps=10,
        num_warmup_steps=5,
        use_mixed_precision=True,
    )
