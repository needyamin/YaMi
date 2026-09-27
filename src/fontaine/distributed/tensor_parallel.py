"""Tensor-parallel matmul.

Column-parallel outputs are concatenated across ranks. Row-parallel partial
outputs are summed. The pure functions below are what each rank computes;
``TensorParallelLinear`` calls them and uses torch.distributed when the
tensor-parallel size is greater than one.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


def column_parallel(x: torch.Tensor, shards: list[torch.Tensor]) -> torch.Tensor:
    """``shards`` are ``[out/world, in]`` pieces of one weight. Result matches ``F.linear``."""
    parts = [F.linear(x, shard) for shard in shards]
    return torch.cat(parts, dim=-1)


def row_parallel(x: torch.Tensor, shards: list[torch.Tensor]) -> torch.Tensor:
    """``shards`` are ``[out, in/world]`` pieces. ``x`` is split on the last dimension."""
    pieces = x.split(shards[0].shape[1], dim=-1)
    if len(pieces) != len(shards):
        raise RuntimeError(
            f"row-parallel input width {x.shape[-1]} does not match "
            f"{len(shards)} shards of width {shards[0].shape[1]}"
        )
    total = F.linear(pieces[0], shards[0])
    for piece, shard in zip(pieces[1:], shards[1:], strict=True):
        total = total + F.linear(piece, shard)
    return total


def split_sequence(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if x.shape[1] % world != 0:
        raise RuntimeError(
            f"sequence length {x.shape[1]} is not divisible by sequence_parallel_size {world}"
        )
    return x.chunk(world, dim=1)[rank]


def gather_sequence(parts: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(parts, dim=1)


class TensorParallelLinear(nn.Module):
    """One rank's slice of a linear layer.

    ``style='column'`` gathers outputs. ``style='row'`` sums partial outputs.
    Both require an initialized process group when ``world_size > 1``.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        style: str,
        rank: int,
        world_size: int,
    ) -> None:
        super().__init__()
        if style not in ("column", "row"):
            raise ValueError(f"tensor parallel style must be 'column' or 'row', got {style!r}")
        if world_size < 2:
            raise ValueError("TensorParallelLinear is for world_size >= 2; use nn.Linear otherwise")
        out, inn = weight.shape
        if style == "column":
            if out % world_size != 0:
                raise RuntimeError(f"column parallel requires out_features {out} divisible by {world_size}")
            local = weight.detach().chunk(world_size, dim=0)[rank].contiguous()
            local_bias = None if bias is None else bias.detach().chunk(world_size, dim=0)[rank].contiguous()
        else:
            if inn % world_size != 0:
                raise RuntimeError(f"row parallel requires in_features {inn} divisible by {world_size}")
            local = weight.detach().chunk(world_size, dim=1)[rank].contiguous()
            local_bias = None if bias is None else bias.detach().contiguous()
            if rank != 0:
                local_bias = None
        self.weight = nn.Parameter(local)
        self.bias = nn.Parameter(local_bias) if local_bias is not None else None
        self.style = style
        self.rank = rank
        self.world_size = world_size
        self.process_group = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "tensor_parallel_size > 1 needs an initialized torch.distributed "
                "process group. Launch with torchrun. Training was not started."
            )
        group = self.process_group
        bias = self.bias if self.bias is not None else x.new_empty(0)
        if self.style == "column":
            return _ColumnParallel.apply(x, self.weight, bias, self.rank, self.world_size, group)
        return _RowParallel.apply(x, self.weight, bias, self.rank, self.world_size, group)


def split_sequence_features(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    if x.shape[-1] % world != 0:
        raise RuntimeError(
            f"feature width {x.shape[-1]} is not divisible by tensor_parallel_size {world}"
        )
    return x.chunk(world, dim=-1)[rank]


def _all_reduce(tensor: torch.Tensor, group) -> torch.Tensor:
    out = tensor.contiguous().clone()
    if group is None:
        dist.all_reduce(out)
    else:
        dist.all_reduce(out, group=group)
    return out


def _all_gather_last(tensor: torch.Tensor, world: int, group) -> torch.Tensor:
    parts = [torch.empty_like(tensor) for _ in range(world)]
    if group is None:
        dist.all_gather(parts, tensor.contiguous())
    else:
        dist.all_gather(parts, tensor.contiguous(), group=group)
    return torch.cat(parts, dim=-1)


class _ColumnParallel(torch.autograd.Function):
    """Replicated input, sharded output features, gathered so every rank sees the full vector."""

    @staticmethod
    def forward(ctx, x, weight, bias, rank: int, world: int, group):  # type: ignore[override]
        ctx.save_for_backward(x, weight, bias)
        ctx.rank = rank
        ctx.world = world
        ctx.group = group
        ctx.has_bias = bias.numel() > 0
        local_bias = bias if ctx.has_bias else None
        local = F.linear(x, weight, local_bias)
        return _all_gather_last(local, world, group)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        x, weight, _bias = ctx.saved_tensors
        local_grad = grad.chunk(ctx.world, dim=-1)[ctx.rank]
        grad_input = _all_reduce(local_grad.matmul(weight), ctx.group)
        flat_grad = local_grad.reshape(-1, local_grad.shape[-1])
        flat_x = x.reshape(-1, x.shape[-1])
        grad_weight = flat_grad.transpose(0, 1).matmul(flat_x)
        grad_bias = flat_grad.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None, None


class _RowParallel(torch.autograd.Function):
    """Sharded input features, partial outputs summed across the tensor-parallel group."""

    @staticmethod
    def forward(ctx, x, weight, bias, rank: int, world: int, group):  # type: ignore[override]
        ctx.save_for_backward(x, weight, bias)
        ctx.rank = rank
        ctx.world = world
        ctx.group = group
        ctx.has_bias = bias.numel() > 0
        local_in = split_sequence_features(x, rank, world)
        local_bias = bias if ctx.has_bias else None
        partial = F.linear(local_in, weight, local_bias)
        return _all_reduce(partial, group)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        x, weight, _bias = ctx.saved_tensors
        local_in = split_sequence_features(x, ctx.rank, ctx.world)
        grad_local_in = grad.matmul(weight)
        grad_input = _all_gather_last(grad_local_in.contiguous(), ctx.world, ctx.group)
        flat_grad = grad.reshape(-1, grad.shape[-1])
        flat_in = local_in.reshape(-1, local_in.shape[-1])
        grad_weight = flat_grad.transpose(0, 1).matmul(flat_in)
        grad_bias = flat_grad.sum(dim=0) if ctx.has_bias else None
        return grad_input, grad_weight, grad_bias, None, None, None


def apply_tensor_parallel(model: nn.Module, rank: int, world_size: int) -> nn.Module:
    """Replace attention and feed-forward linears with rank-local shards.

    The router stays replicated. Call this only after the process group exists
    and ``world_size`` matches it.
    """
    if world_size < 2:
        return model
    for module in model.modules():
        for name, style in (
            ("q_proj", "column"),
            ("k_proj", "column"),
            ("v_proj", "column"),
            ("gate_proj", "column"),
            ("up_proj", "column"),
            ("fc", "column"),
            ("out_proj", "row"),
            ("down_proj", "row"),
            ("proj", "row"),
        ):
            layer = getattr(module, name, None)
            if isinstance(layer, nn.Linear):
                bias = layer.bias.data if layer.bias is not None else None
                replacement = TensorParallelLinear(
                    layer.weight.data, bias, style, rank, world_size
                )
                setattr(module, name, replacement)
    return model
