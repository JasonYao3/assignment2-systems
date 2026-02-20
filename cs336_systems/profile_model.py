import logging
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM, scaled_dot_product_attention
from einops import einsum
from cs336_basics.nn_utils import softmax
import math


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    with nvtx.range("computing softmax"):
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


# cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def profile_forward(model, input_data):
    """
    Profiles the forward pass of the model.
    """
    with nvtx.range("forward"):
        output = model(input_data)


def profile_forward_backward(model, input_data):
    """
    Profiles the forward and backward pass of the model.
    """
    model.zero_grad()
    with nvtx.range("forward"):
        output = model(input_data)

    with nvtx.range("backward"):
        loss = output.sum()
        loss.backward()


if __name__ == "__main__":
    # Define hyperparameters (use a small model to start)
    vocab_size = 10000
    batch_size = 4
    context_length = 128
    d_model = 768
    num_layers = 12
    num_heads = 12
    d_ff = 3072
    rope_theta = 10000.0

    print("Initializing model...")
    # NOTE: nsys only works with NVIDIA GPUs. MPS (Mac) profiling uses a different tool (Instruments/Metal).
    # Since you are on a Mac, you won't be able to run `nsys`.
    # If the assignment requires Nsight, you MUST run this code on the provided Slurm cluster / NVIDIA GPU.

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Profiling on {device}")

    model = BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta).to(device)

    print("Generating dummy data...")
    input_data = torch.randint(vocab_size, (batch_size, context_length)).to(device)

    print("Warming up...")
    for _ in range(3):
        profile_forward(model, input_data)
        profile_forward_backward(model, input_data)

    print("Running profiling loops...")
    with nvtx.range("Measurement Loop"):
        for _ in range(5):
            profile_forward(model, input_data)
            profile_forward_backward(model, input_data)

    print("Profiling complete.")
