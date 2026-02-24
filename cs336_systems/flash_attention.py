import torch
import math


class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        """
        Pure PyTorch implementation of FlashAttention-2 forward pass.

        Q: (batch_size, n_heads, seq_len, d_head)
        K: (batch_size, n_heads, seq_len, d_head)
        V: (batch_size, n_heads, seq_len, d_head)
        """
        # Determine your own tile sizes (must be clean powers of 2 and >= 16)
        B_q = 16
        B_k = 16

        seq_len = Q.shape[-2]
        d = Q.shape[-1]

        # We need to maintain all the preceding batch dimensions
        batch_dims = Q.shape[:-2]

        # Initialize output O, logsumexp L, and running max m
        O = torch.zeros_like(Q)
        L = torch.zeros((*batch_dims, seq_len), device=Q.device)

        # TO DO: Implement FlashAttention-2 Forward Pass (Algorithm 1)
        # 1. Split Q into T_q tiles of size B_q
        T_q = math.ceil(seq_len / B_q)
        # 2. Split K, V into T_k tiles of size B_k
        T_k = math.ceil(seq_len / B_k)

        # 3. Outer loop over Q tiles (i = 1 ... T_q)
        for i in range(T_q):
            start_i = i * B_q
            end_i = min(start_i + B_q, seq_len)

            Q_i = Q[..., start_i:end_i, :]

            # Initialize states for this Q block
            m_i = torch.full(
                (*batch_dims, (end_i - start_i), 1),
                float("-inf"),
                device=Q.device,
            )
            l_i = torch.zeros((*batch_dims, (end_i - start_i), 1), device=Q.device)
            O_i = torch.zeros((*batch_dims, (end_i - start_i), d), device=Q.device)

            # 4. Inner loop over K,V tiles (j = 1 ... T_k)
            for j in range(T_k):
                start_j = j * B_k
                end_j = min(start_j + B_k, seq_len)

                K_j = K[..., start_j:end_j, :]
                V_j = V[..., start_j:end_j, :]

                # Equation 4: Compute tile of attention scores (S_ij)
                # Outer dims are batched, so we just tranpose the last two (d_head and seq_len_chunk)
                S_ij = (Q_i @ K_j.transpose(-2, -1)) / math.sqrt(d)

                # Check for Causal Masking
                if is_causal:
                    # Create a boolean mask where Query index < Key index
                    q_idx = torch.arange(start_i, end_i, device=Q.device).unsqueeze(1)
                    k_idx = torch.arange(start_j, end_j, device=Q.device).unsqueeze(0)
                    mask = q_idx < k_idx
                    S_ij = torch.where(mask, float("-1e6"), S_ij)

                # Find the maximum value in the row for safe mathematical stable softmax
                row_max = S_ij.max(dim=-1, keepdim=True).values

                # Equation 5: Update the running max m_new
                m_new = torch.maximum(m_i, row_max)

                # Equation 6: Compute the exponentiated unnormalized scores P_tilde
                P_tilde = torch.exp(S_ij - m_new)

                # Equation 7: Update our running denominator sum l_new
                # Scale the previous sum by the difference in our maxes to keep it mathematically sound
                l_new = torch.exp(m_i - m_new) * l_i + P_tilde.sum(dim=-1, keepdim=True)

                # Equation 8: Update our output block O_i
                O_i = torch.exp(m_i - m_new) * O_i + (P_tilde @ V_j)

                # Set m_i and l_i for the next loop iteration!
                m_i = m_new
                l_i = l_new

            # --- OUTSIDE INNER LOOP ---

            # Equation 9: Normalize the output by the final denominator
            O_i = O_i / l_i

            # Equation 10: Calculate the Log Sum Exp mathematically
            L_i = m_i + torch.log(l_i)

            # Write O_i into the final output matrix
            O[..., start_i:end_i, :] = O_i

            # Write L_i into the L matrix (removing the 1 dim at the end to match shape)
            L[..., start_i:end_i] = L_i.squeeze(-1)

        # Save for backward pass
        ctx.is_causal = is_causal
        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError("Backward pass not implemented yet")
