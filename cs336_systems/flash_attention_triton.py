import triton
import triton.language as tl
import torch
import math


@triton.jit
def flash_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    L_ptr,
    stride_qb,
    stride_qq,
    stride_qd,
    stride_kb,
    stride_kk,
    stride_kd,
    stride_vb,
    stride_vk,
    stride_vd,
    stride_ob,
    stride_oq,
    stride_od,
    stride_lb,
    stride_lq,
    N_QUERIES,
    N_KEYS,
    scale,
    is_causal: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    """
    Triton Kernel for FlashAttention-2 Forward Pass.
    """
    # 1. Get the program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 2. Offset each pointer with the corresponding batch index * batch stride
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(D, N_KEYS),
        strides=(stride_kd, stride_kk),  # We transpose K by swapping strides!
        offsets=(0, 0),
        block_shape=(D, K_TILE_SIZE),
        order=(0, 1),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # L is a 1D tensor (per batch), so we don't need a block pointer, just normal pointer arithmetic
    l_ptrs = L_ptr + batch_index * stride_lb + query_tile_index * Q_TILE_SIZE * stride_lq + tl.arange(0, Q_TILE_SIZE)

    # 3. Initialize state in SRAM
    m_i = tl.full([Q_TILE_SIZE], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    # Load Q
    q = tl.load(Q_block_ptr)

    # 4. Loop over K, V tiles
    for j in range(0, tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load the current K and V tiles
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        # Equation 4: Compute tile of attention scores (s_ij)
        # Note: k is already transposed due to strides swapping setup in blockptr
        s_ij = tl.dot(q, k) * scale

        # Check for Causal Masking
        if is_causal:
            # Create a 2D integer matrix of our indices
            q_offset = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offset = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

            # Using none broadcasting creates a (Q_TILE_SIZE, K_TILE_SIZE) boolean mask
            mask = q_offset[:, None] >= k_offset[None, :]
            s_ij = tl.where(mask, s_ij, float("-1e6"))

        # Equation 5: Update the running max m_new
        row_max = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, row_max)

        # Equation 6: Compute the exponentiated unnormalized scores P_tilde
        p_tilde = tl.exp(s_ij - m_new[:, None])

        # Equation 7: Update our running denominator sum l_new
        # Scale the old l_i by the diff in maxes, then add the new sum
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p_tilde, axis=1)

        # Equation 8: Update our output block output block
        # We must cast the float32 p_tilde down to the precision of V before doing dot-product
        acc = acc * alpha[:, None] + tl.dot(p_tilde.to(V_ptr.type.element_ty), v)

        # Update running max for the next loop iteration
        m_i = m_new

        # Advance block pointers for next iteration
        K_block_ptr = tl.advance(K_block_ptr, (0, K_TILE_SIZE))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    # 5. After the loop, calculate L_i and write to output pointers
    # Equation 10: Calculate L_i
    L_i = m_i + tl.math.log(l_i)
    tl.store(l_ptrs, L_i)

    # Equation 9: Normalize the output by the final denominator and store
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Q_ptr.type.element_ty))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Flatten batch and heads into a single batch dimension for Triton
        batch_size, n_heads, seq_len, d = Q.shape
        q = Q.view(batch_size * n_heads, seq_len, d)
        k = K.view(batch_size * n_heads, seq_len, d)
        v = V.view(batch_size * n_heads, seq_len, d)

        o = torch.empty_like(q)
        l = torch.empty((batch_size * n_heads, seq_len), device=q.device, dtype=torch.float32)

        B_q = 16
        B_k = 16

        # Launch 2D grid: (number of Q blocks, batch_size * n_heads)
        grid = (triton.cdiv(seq_len, B_q), batch_size * n_heads)

        flash_fwd_kernel[grid](
            q,
            k,
            v,
            o,
            l,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            l.stride(0),
            l.stride(1),
            N_QUERIES=seq_len,
            N_KEYS=seq_len,
            scale=1.0 / math.sqrt(d),
            is_causal=is_causal,
            D=d,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
        )

        ctx.save_for_backward(l.view(batch_size, n_heads, seq_len), Q, K, V, o.view(Q.shape))
        ctx.is_causal = is_causal
        return o.view(Q.shape)

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Triton Backward pass not implemented yet")
