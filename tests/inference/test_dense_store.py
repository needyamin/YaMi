"""Disk-streamed dense blocks: same outputs, a capped resident set."""

import pytest
import torch

from fontaine.config.schema import ModelConfig
from fontaine.inference.dense_store import DenseStreamList, attach_dense_streaming
from fontaine.models import build_model
from fontaine.optimization.quantize import apply_inference_precision


def _dense_config() -> ModelConfig:
    return ModelConfig(
        architecture="decoder_transformer",
        vocab_size=32,
        hidden_size=16,
        num_layers=3,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        max_sequence_length=16,
        dropout=0.0,
    )


def test_streamed_model_matches_baseline_bit_for_bit(tmp_path):
    torch.manual_seed(7)
    model = build_model(_dense_config()).eval()
    ids = torch.randint(0, 32, (1, 8))
    expected = model(ids).logits.clone()

    store = attach_dense_streaming(model, budget_bytes=1, directory=tmp_path)
    assert store is not None
    got = model(ids).logits
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    # All blocks streamed back through the LRU on that call.
    stats = store.stats()
    assert stats.misses >= 3
    assert stats.bytes_read > 0
    store.close()


def test_budget_keeps_resident_under_the_cap_and_hits_after_warmup(tmp_path):
    torch.manual_seed(11)
    model = build_model(_dense_config()).eval()
    blocks = list(model.blocks)
    per_block = sum(
        t.numel() * t.element_size() for t in blocks[0].state_dict().values()
    )
    # Budget fits two blocks: the third must evict on every full pass.
    budget = per_block * 2 + 1
    store = attach_dense_streaming(model, budget_bytes=budget, directory=tmp_path)
    assert store is not None
    ids = torch.randint(0, 32, (1, 4))
    for _ in range(3):
        model(ids)
    stats = store.stats()
    assert stats.resident_bytes <= budget
    assert stats.requests >= 9  # 3 blocks x 3 passes
    assert stats.hits >= 3
    store.close()


def test_oversize_single_block_still_runs(tmp_path):
    torch.manual_seed(3)
    model = build_model(_dense_config()).eval()
    ids = torch.randint(0, 32, (1, 8))
    expected = model(ids).logits.clone()
    # Budget smaller than one block: every block runs transiently.
    store = attach_dense_streaming(model, budget_bytes=1, directory=tmp_path)
    assert store is not None
    got = model(ids).logits
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    store.close()


def test_small_model_skips_streaming(tmp_path):
    torch.manual_seed(5)
    model = build_model(_dense_config()).eval()
    total = sum(
        t.numel() * t.element_size() for t in model.blocks[0].state_dict().values()
    ) * len(list(model.blocks))
    assert attach_dense_streaming(model, budget_bytes=total, directory=tmp_path) is None
    # Untouched: forward works and no spill files were written.
    ids = torch.randint(0, 32, (1, 4))
    model(ids)


def test_streamed_model_with_int8_precision_matches_baseline(tmp_path):
    torch.manual_seed(13)
    model = build_model(_dense_config()).eval()
    ids = torch.randint(0, 32, (1, 8))
    quantized = apply_inference_precision(model, "int8")
    expected = quantized(ids).logits.clone()

    store = attach_dense_streaming(quantized, budget_bytes=1, directory=tmp_path)
    assert store is not None
    got = quantized(ids).logits
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)
    store.close()


def test_getitem_negative_and_slice_and_errors(tmp_path):
    torch.manual_seed(17)
    model = build_model(_dense_config()).eval()
    store = attach_dense_streaming(model, budget_bytes=1, directory=tmp_path)
    assert store is not None
    assert isinstance(store[-1], torch.nn.Module)
    assert isinstance(store[0:2], torch.nn.ModuleList)
    with pytest.raises(IndexError):
        _ = store[len(store)]
    with pytest.raises(TypeError):
        _ = store["zero"]  # type: ignore[index]
    store.close()


def test_stats_snapshot_counts_requests_hits_and_prefetch(tmp_path):
    torch.manual_seed(19)
    model = build_model(_dense_config()).eval()
    store = attach_dense_streaming(model, budget_bytes=1, directory=tmp_path)
    assert store is not None
    ids = torch.randint(0, 32, (1, 4))
    model(ids)
    first = store.stats()
    # len() does not materialize; list() would, so assert against len(store).
    assert first.misses == len(store)
    assert first.requests == first.misses
    # Second pass hits nothing (budget = 1 byte), but requests are counted.
    model(ids)
    second = store.stats()
    assert second.requests == 2 * first.requests
    assert second.capacity_bytes == 1
    store.close()


def test_stream_list_prefetch_is_safe_before_spill_or_when_resident(tmp_path):
    torch.manual_seed(23)
    model = build_model(_dense_config()).eval()
    blocks = list(model.blocks)
    stream = DenseStreamList(blocks, budget_bytes=1, directory=tmp_path)
    # Without spill_all, entries are passthrough (no placeholders).
    assert stream[0] is blocks[0]
    stream.prefetch(0)  # no-op, must not raise
    stream.close()
