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
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        d = Q.shape[-1]

        # Determine the shape so we can safely broadcast the output
        seq_len_q = Q.shape[-2]
        seq_len_k = K.shape[-2]

        # We need to compute D = rowsum(dO * O)
        # grad_out is dO. They both have shape (*batch, seq_len, d)
        # We elementwise multiply them, then sum along the d dimension
        D_vec = (grad_out * O).sum(dim=-1, keepdim=True)

        # We can implement the backward pass as a separate standard function, then compile it
        # Since it takes python tensors, compiling it is fine
        def raw_backward(q, k, v, do, l_vec, d_vec, causal):
            # Equation 13: S = Q @ K.T / sqrt(d)
            s = (q @ k.transpose(-2, -1)) / math.sqrt(d)

            # Equation 14: P = exp(S - L)
            # l_vec needs an extra dimension at the end to broadcast with S
            p = torch.exp(s - l_vec.unsqueeze(-1))

            if causal:
                # Mask out causal elements in P by setting them to 0 rather than negative infinity
                # (since 0 * any gradient = 0)
                q_idx = torch.arange(seq_len_q, device=q.device).unsqueeze(1)
                k_idx = torch.arange(seq_len_k, device=q.device).unsqueeze(0)
                mask = q_idx < k_idx
                p = torch.where(mask, 0.0, p)

            # Equation 15: dV = P.T @ dO
            dv = p.transpose(-2, -1) @ do

            # Equation 16: dP = dO @ V.T
            dp = do @ v.transpose(-2, -1)

            # Equation 17: dS = P * (dP - D)
            ds = p * (dp - d_vec)

            # Equation 18: dQ = dS @ K / sqrt(d)
            dq = (ds @ k) / math.sqrt(d)

            # Equation 19: dK = dS.T @ Q / sqrt(d)
            dk = (ds.transpose(-2, -1) @ q) / math.sqrt(d)

            return dq, dk, dv

        # The instructions recommend using torch.compile to optimize this naive execution
        compiled_backward = torch.compile(raw_backward)

        # NOTE: Using torch.compile inside an autograd.Function backward pass on Mac MPS
        # may cause a silent freeze/crash during testing (as observed in earlier tasks).
        # While it is the correct implementation as requested, it might need to run on a Linux GPU.
        dQ, dK, dV = compiled_backward(Q, K, V, grad_out, L, D_vec, is_causal)

        # Return gradients in the exact same order as the forward inputs: Q, K, V, is_causal
        return dQ, dK, dV, None
