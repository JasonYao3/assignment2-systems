import torch
import timeit
import itertools
import math
from cs336_basics.nn_utils import softmax
from einops import einsum


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Standard PyTorch attention implementation (no multihead).
    """
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    attention_weights = softmax(attention_scores, dim=-1)
    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def get_memory_info(device):
    """Returns memory allocated in MB for CUDA or MPS."""
    if device == "cuda":
        return torch.cuda.memory_allocated() / (1024**2)
    elif device == "mps":
        return torch.mps.current_allocated_memory() / (1024**2)
    return 0


def benchmark_attention():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Benchmarking on: {device.upper()}\n")

    batch_size = 8
    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    print(f"{'d_model':<10} | {'seq_len':<10} | {'Fwd (ms)':<10} | {'Bwd (ms)':<10} | {'Mem (MB)':<10} | {'Status'}")
    print("-" * 75)

    for d in d_models:
        for s in seq_lens:
            Q = torch.randn(batch_size, s, d, device=device, requires_grad=True)
            K = torch.randn(batch_size, s, d, device=device, requires_grad=True)
            V = torch.randn(batch_size, s, d, device=device, requires_grad=True)

            try:
                # Warmup (5 times) and Time Forward Pass (100 times)
                for _ in range(5):
                    scaled_dot_product_attention(Q, K, V)

                torch.mps.synchronize()
                start_fwd = timeit.default_timer()
                for i in range(100):
                    scaled_dot_product_attention(Q, K, V)

                torch.mps.synchronize()
                end_fwd = timeit.default_timer()
                fwd_time = ((end_fwd - start_fwd) / 100) * 1000

                # Get peak memory
                mem_mb = get_memory_info(device)

                # Time Backward Pass (100 times)
                start_bwd = timeit.default_timer()

                for i in range(100):
                    output = scaled_dot_product_attention(Q, K, V)
                    loss = output.sum()
                    loss.backward()

                    Q.grad = None
                    K.grad = None
                    V.grad = None
                end_bwd = timeit.default_timer()
                bwd_time = ((end_bwd - start_bwd) / 100) * 1000

                # --- COMPILED ATTENTION BENCHMARK ---
                compiled_attention = torch.compile(scaled_dot_product_attention)

                # Warmup (5 times) for compiled pass
                for _ in range(5):
                    compiled_attention(Q, K, V)

                torch.mps.synchronize()
                start_fwd_comp = timeit.default_timer()
                for i in range(100):
                    compiled_attention(Q, K, V)

                torch.mps.synchronize()
                end_fwd_comp = timeit.default_timer()
                fwd_time_comp = ((end_fwd_comp - start_fwd_comp) / 100) * 1000

                # Get peak memory
                mem_mb_comp = get_memory_info(device)

                # Time Backward Pass (100 times)
                start_bwd_comp = timeit.default_timer()

                for i in range(100):
                    output = compiled_attention(Q, K, V)
                    loss = output.sum()
                    loss.backward()

                    Q.grad = None
                    K.grad = None
                    V.grad = None
                end_bwd_comp = timeit.default_timer()
                bwd_time_comp = ((end_bwd_comp - start_bwd_comp) / 100) * 1000

                print(f"{d:<10} | {s:<10} | {fwd_time:<10.3f} | {bwd_time:<10.3f} | {mem_mb:<10.2f} | {'Ok'}")
                print(
                    f"{d:<10} | {s:<10} | {fwd_time_comp:<10.3f} | {bwd_time_comp:<10.3f} | {mem_mb_comp:<10.2f} | {'JIT Compiled'}"
                )
                print("-" * 75)

            except RuntimeError as e:
                # If memory explodes, PyTorch throws a RuntimeError.
                print(f"{d:<10} | {s:<10} | {'OOM':<10} | {'OOM':<10} | {'OOM':<10} | {'Failed'}")


if __name__ == "__main__":
    benchmark_attention()
