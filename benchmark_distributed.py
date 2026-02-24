import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import pandas as pd


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_benchmark(rank, world_size, sizes_mb, backend, device_type):
    setup(rank, world_size, backend)

    if device_type == "cuda":
        # Pin the process to its specific GPU
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"

    results = []

    # 5 iterations of warmup as recommended by the PDF
    warmup_steps = 5
    # Number of measured iterations
    measure_steps = 10

    for size_mb in sizes_mb:
        # 1 float32 = 4 bytes. 1 MB = 1024 * 1024 bytes.
        # Number of elements = (MB * 1024 * 1024) / 4
        num_elements = int((size_mb * 1024 * 1024) / 4)

        # We need a list to hold the recorded times for this worker
        times = []

        for step in range(warmup_steps + measure_steps):
            # Generate the random tensor for this rank
            data = torch.rand(num_elements, dtype=torch.float32, device=device)

            # Sync before starting timer if using GPU
            if device_type == "cuda":
                torch.cuda.synchronize(device)

            # Use high resolution timer
            start_time = time.perf_counter()

            # Perform the collective communication
            dist.all_reduce(data, async_op=False)

            # Ensure the GPU has actually finished the work before stopping timer
            if device_type == "cuda":
                torch.cuda.synchronize(device)

            end_time = time.perf_counter()

            # Only record after warmup
            if step >= warmup_steps:
                # Convert to milliseconds
                times.append((end_time - start_time) * 1000)

        # Calculate the average time taken for this tensor size on this rank
        avg_time = sum(times) / measure_steps

        # We want to gather the average time from ALL ranks to report an overall number.
        # We can construct a 1-element tensor and use all_reduce to sum them, then divide by world_size!
        avg_time_tensor = torch.tensor([avg_time], dtype=torch.float32, device=device)
        dist.all_reduce(avg_time_tensor, op=dist.ReduceOp.SUM)

        overall_avg_time_ms = avg_time_tensor[0].item() / world_size

        # Only have rank 0 print/save the results to avoid duplicate logs
        if rank == 0:
            results.append(
                {
                    "Backend": backend,
                    "Device": device_type,
                    "World Size": world_size,
                    "Tensor Size (MB)": size_mb,
                    "All-Reduce Ms": f"{overall_avg_time_ms:.4f}",
                }
            )

    if rank == 0:
        # Convert to DataFrame just so it prints out nicely for the user
        df = pd.DataFrame(results)
        print(df.to_markdown(index=False))
        print("\n")

        # We will append to a markdown file so multiple runs append together
        with open("distributed_benchmark_results.md", "a") as f:
            f.write(df.to_markdown(index=False))
            f.write("\n\n")

    cleanup()


def main():
    sizes_mb = [1, 10, 100, 1024]

    # Optional parameters. The user can tweak these on their cluster.
    # Notice: Mac only supports gloo/cpu, you will need a GPU box for nccl/cuda runs
    configs = [
        # (world_size, backend, device_type)
        (2, "gloo", "cpu"),
        (4, "gloo", "cpu"),
        # To run these below, uncomment them on your H100 Server
        # (2, "nccl", "cuda"),
        # (4, "nccl", "cuda"),
        # (6, "nccl", "cuda"),
    ]

    print("Starting Distributed Benchmarks...")
    for world_size, backend, device_type in configs:
        print(f"\n--- Testing {world_size} Processes using {backend.upper()} on {device_type.upper()} ---")
        # Ensure our local machine actually has enough GPUs if requesting CUDA
        if device_type == "cuda" and torch.cuda.device_count() < world_size:
            print(f"Skipping: Node only has {torch.cuda.device_count()} GPUs, but config requires {world_size}.")
            continue

        # Spawn `world_size` processes, each running the `run_benchmark` function
        mp.spawn(
            run_benchmark,
            args=(world_size, sizes_mb, backend, device_type),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
