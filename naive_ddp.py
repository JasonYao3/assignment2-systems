import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5))

    def forward(self, x):
        return self.net(x)


def setup(rank, world_size):
    """Initialize the process group"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # Using gloo for CPU compatibility on local machines
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    """Destroy the process group"""
    dist.destroy_process_group()


def train_naive_ddp(rank, world_size):
    """
    Naive DDP implementation:
    1. Replicate model on all ranks (using identical seeds for random initialization)
    2. Split data batch across ranks
    3. Forward pass + Backward pass to calculate local gradients
    4. Loop over all parameters and manually all_reduce their gradients
    5. Optimizer step!
    """
    setup(rank, world_size)
    print(f"[Rank {rank}] Initialized.")

    # 1. Setup the identical model on all ranks
    # Fixing the seed ensures that all 4 processes start with the exact same initialized weights.
    # In a real cluster, you would usually have Rank 0 broadcast its random weights to everyone instead.
    torch.manual_seed(42)
    device = torch.device("cpu")  # Replace with 'cuda' on GPU server
    model = ToyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Generate some toy data
    # Shape: (batch_size=100, features=10)
    torch.manual_seed(100)  # Reset seed so data isn't same as weights
    data = torch.randn(100, 10).to(device)
    targets = torch.randn(100, 5).to(device)
    loss_fn = nn.MSELoss()

    epochs = 3
    # 2. Split the batch
    # We carve out a specific slice of the data for this specific worker to process!
    batch_size_per_rank = len(data) // world_size
    start_idx = rank * batch_size_per_rank
    end_idx = start_idx + batch_size_per_rank

    local_data = data[start_idx:end_idx]
    local_targets = targets[start_idx:end_idx]

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 3. Forward Pass & Backward Pass (Local compute)
        start_time = time.time()
        outputs = model(local_data)
        loss = loss_fn(outputs, local_targets)

        # Calculate gradients using only this rank's specific slice of data
        loss.backward()

        # 4. NAIVE DDP: Gradient Synchronization
        # We loop through every single parameter tensor linearly.
        # We average its gradient across all nodes via all_reduce.
        for param in model.parameters():
            if param.grad is not None:
                # all_reduce expects the tensor itself. It sums them up across all processes in-place!
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                # To get the average, we just divide by the world size
                param.grad.data /= world_size

        # 5. Optimizer Step
        # Everyone now has the identically averaged gradients, so taking a step ensures weights stay identical!
        optimizer.step()

        end_time = time.time()
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Time: {(end_time - start_time) * 1000:.2f}ms")

    if rank == 0:
        print("\nNaive DDP Training Complete!")

    cleanup()


if __name__ == "__main__":
    world_size = 4
    print(f"Starting Naive DDP Training Simulation with {world_size} processes...")
    try:
        mp.spawn(train_naive_ddp, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Failed to spawn processes: {e}")
