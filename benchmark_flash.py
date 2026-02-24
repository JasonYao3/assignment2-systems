import torch
import triton
import pandas as pd
import itertools

from torch.nn.functional import scaled_dot_product_attention

try:
    from cs336_systems.flash_attention_triton import FlashAttention2Triton
except ImportError:
    FlashAttention2Triton = None

from cs336_systems.flash_attention import FlashAttention2


def run_benchmark():
    if not torch.cuda.is_available():
        print("CUDA not available. Benchmarking requires a GPU (preferably H100). Exiting.")
        return

    if FlashAttention2Triton is None:
        print("Triton not installed. Benchmarking requires Triton on Linux/NVIDIA. Exiting.")
        return

    # Sweep settings defined in the problem description
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_heads = [16, 32, 64, 128]
    dtypes = [torch.float32, torch.bfloat16]
    batch_size = 1
    n_heads = 16  # Not swept, but standard

    results = []

    print(f"Running benchmarks... This will take a while.")

    # Pre-compile the backward pass as instructed
    compiled_flash_autograd = torch.compile(FlashAttention2.apply)

    for seq_len, d_head, dtype in itertools.product(seq_lens, d_heads, dtypes):
        print(f"Benchmarking (seq_len={seq_len}, d_head={d_head}, dtype={dtype})...")

        try:
            # Create random inputs
            q = torch.randn(
                batch_size,
                n_heads,
                seq_len,
                d_head,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            k = torch.randn(
                batch_size,
                n_heads,
                seq_len,
                d_head,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )
            v = torch.randn(
                batch_size,
                n_heads,
                seq_len,
                d_head,
                device="cuda",
                dtype=dtype,
                requires_grad=True,
            )

            # The backward pass requires incoming gradients
            do = torch.randn_like(q)

            # -------------------------------------------------------------
            # PyTorch Baselines
            # -------------------------------------------------------------
            def pytorch_fwd():
                return scaled_dot_product_attention(q, k, v, is_causal=True)

            def pytorch_bwd():
                out = scaled_dot_product_attention(q, k, v, is_causal=True)
                out.backward(do, retain_graph=True)

            def pytorch_fwd_bwd():
                out = scaled_dot_product_attention(q, k, v, is_causal=True)
                out.backward(do, retain_graph=True)

            # -------------------------------------------------------------
            # Custom FlashAttention-2
            # -------------------------------------------------------------
            def flash_fwd():
                return FlashAttention2Triton.apply(q, k, v, True)

            def flash_bwd():
                # For backward, we use the compiled PyTorch version as instructed
                out = compiled_flash_autograd(q, k, v, True)
                out.backward(do, retain_graph=True)

            def flash_fwd_bwd():
                out = compiled_flash_autograd(q, k, v, True)
                out.backward(do, retain_graph=True)

            # Perform warmup and benchmarking via Triton's built-in tool
            # (Note: For massive matrices, PyTorch might OOM here, Triton bench handles catching some of these)
            ms_pt_fwd = triton.testing.do_bench(pytorch_fwd, quantiles=[0.5])[0]
            ms_pt_bwd = triton.testing.do_bench(pytorch_bwd, quantiles=[0.5])[0]
            ms_pt_both = triton.testing.do_bench(pytorch_fwd_bwd, quantiles=[0.5])[0]

            ms_flash_fwd = triton.testing.do_bench(flash_fwd, quantiles=[0.5])[0]
            ms_flash_bwd = triton.testing.do_bench(flash_bwd, quantiles=[0.5])[0]
            ms_flash_both = triton.testing.do_bench(flash_fwd_bwd, quantiles=[0.5])[0]

            results.append(
                {
                    "seq_len": seq_len,
                    "d_head": d_head,
                    "dtype": str(dtype).split(".")[-1],
                    "pt_fwd_ms": f"{ms_pt_fwd:.4f}",
                    "pt_bwd_ms": f"{ms_pt_bwd:.4f}",
                    "pt_both_ms": f"{ms_pt_both:.4f}",
                    "flash_fwd_ms": f"{ms_flash_fwd:.4f}",
                    "flash_bwd_ms": f"{ms_flash_bwd:.4f}",
                    "flash_both_ms": f"{ms_flash_both:.4f}",
                }
            )

        except torch.cuda.OutOfMemoryError:
            results.append(
                {
                    "seq_len": seq_len,
                    "d_head": d_head,
                    "dtype": str(dtype).split(".")[-1],
                    "pt_fwd_ms": "OOM",
                    "pt_bwd_ms": "OOM",
                    "pt_both_ms": "OOM",
                    "flash_fwd_ms": "OOM",
                    "flash_bwd_ms": "OOM",
                    "flash_both_ms": "OOM",
                }
            )
            torch.cuda.empty_cache()
            print(f"  -> OOM encountered at seq_len={seq_len}")
        except Exception as e:
            print(f"  -> Error: {e}")

    # Output to markdown table
    df = pd.DataFrame(results)

    print("\n\n--- Benchmarking Results ---")
    print(df.to_markdown(index=False))

    # Save to a file so it can be easily copied to the writeup
    with open("flash_benchmark_results.md", "w") as f:
        f.write(df.to_markdown(index=False))
    print("\nResults saved to flash_benchmark_results.md")


if __name__ == "__main__":
    run_benchmark()
