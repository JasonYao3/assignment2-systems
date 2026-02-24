import torch
import torch.nn as nn
import torch.distributed as dist


class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        """
        Distributed Data Parallel Wrapper.
        Overlaps gradient communication and backward pass computation.
        """
        super().__init__()
        self.module = module

        # Determine the current rank and world size from the process group
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Keep track of active asynchronous communication handles
        self.communication_handles = []

        # Initial synchronization to ensure identical model weights
        # We broadcast from rank 0 to all other ranks!
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # Register the backward hook to mathematically overlap communication.
        # This hook fires IMMEDIATELY after a parameter's gradient finishes computing,
        # starting the AllReduce process while the rest of the backward pass continues!
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._gradient_sync_hook)

    def _gradient_sync_hook(self, param: torch.Tensor):
        """
        Hook called by PyTorch immediately after a parameter's gradient is accumulated.
        Fires an asynchronous all_reduce directly on the gradient tensor and saves the handle.
        """
        # Average the gradient directly in place!
        # Remember: all_reduce naturally takes sums. To keep the math sound, we usually divide by world_size.
        # So we divide before syncing to avoid numeric overflows or synchronization divergence!
        # (Though we can also divide after, but in asynchronous contexts doing it before is cleaner!)
        if param.grad is not None:
            param.grad.data.div_(self.world_size)
            # async_op=True allows the backward pass to continue computing the next layer immediately!
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.communication_handles.append(handle)

    def forward(self, *inputs, **kwargs):
        """Pass the forward operation straight to the wrapped module"""
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """
        Wait until every single asynchronous all_reduce queued by the backward hooks completes.
        This must be called right before optimizer.step()!
        """
        for handle in self.communication_handles:
            handle.wait()

        # Clear the list for the next batch pass!
        self.communication_handles.clear()
