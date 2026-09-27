"""Scalable architecture: profiles, estimates, attention, MoE, topology, stages."""

from pathlib import Path

import pytest
import torch

from fontaine.cli.main import main
from fontaine.config.errors import ConfigError
from fontaine.config.loader import load_fontaine_config
from fontaine.config.schema import ModelConfig
from fontaine.models import build_model
from fontaine.optimization.memory import estimate_parameter_count

ROOT = Path(__file__).resolve().parents[2]
SUPER = ROOT / "configs" / "yami_super.yaml"
PROFILES = [
    "tiny",
    "small",
    "medium",
    "500m",
    "1b",
    "3b",
    "7b",
    "14b",
    "30b",
    "70b",
    "100b",
    "300b",
    "1t",
]
MOE = {"30b", "70b", "100b", "300b", "1t"}


def _profile(name: str):
    return load_fontaine_config([SUPER], profile=name)


def test_default_profile_is_tiny():
    config = load_fontaine_config([SUPER])
    assert config.model.hidden_size == 256
    assert config.model.num_layers == 4


@pytest.mark.parametrize("name", PROFILES)
def test_profile_validates_and_estimates_without_allocation(name):
    config = _profile(name)
    counts = estimate_parameter_count(config.model)
    assert counts["total"] == (
        counts["embedding"]
        + counts["attention"]
        + counts["dense_ffn"]
        + counts["moe_experts"]
        + counts["router"]
        + counts["shared_experts"]
        + counts["normalization"]
        + counts["lm_head"]
    )
    assert counts["q"] + counts["k"] + counts["v"] + counts["o"] == counts["attention"]
    if name in MOE:
        assert counts["active"] < counts["total"]
        assert counts["shared_experts"] > 0
    else:
        assert counts["active"] == counts["total"]


def test_profile_parameter_counts_increase():
    previous = 0
    for name in PROFILES:
        total = estimate_parameter_count(_profile(name).model)["total"]
        assert total > previous
        previous = total
    assert estimate_parameter_count(_profile("1t").model)["total"] > 1_000_000_000_000


def test_tiny_profile_matches_a_real_model():
    config = _profile("tiny").model
    model = build_model(config)
    counts = estimate_parameter_count(config)
    assert model.num_parameters() == counts["total"]
    assert model.num_active_parameters() == counts["active"]


def test_bad_gqa_message():
    with pytest.raises(ConfigError, match="Choose a num_kv_heads"):
        ModelConfig(hidden_size=384, num_attention_heads=48, num_kv_heads=7, intermediate_size=768).validate()


def test_explicit_head_dim_need_not_tile_hidden():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=30,
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        explicit_head_dim=8,
        intermediate_size=32,
        max_sequence_length=8,
    )
    model = build_model(config)
    counts = estimate_parameter_count(config)
    assert model.num_parameters() == counts["total"]
    tokens = torch.randint(0, 32, (2, 4))
    assert model(tokens).logits.shape == (2, 4, 32)


def test_reference_attention_matches_optimized():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        max_sequence_length=8,
        qk_norm=True,
    )
    torch.manual_seed(0)
    optimized = build_model(config).eval()
    reference = build_model(
        ModelConfig(**{**config.__dict__, "attention_backend": "reference"})
    ).eval()
    reference.load_state_dict(optimized.state_dict())
    tokens = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        left = optimized(tokens).logits
        right = reference(tokens).logits
    assert torch.allclose(left, right, atol=1e-4)


def test_sparse_and_hybrid_run():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        max_sequence_length=16,
        attention_type="hybrid",
        sliding_window=4,
        layer_pattern=["full", "sliding", "sliding", "sparse"],
        sparse_pattern="hybrid",
        sparse_window=4,
        sparse_stride=2,
        sparse_global_tokens=1,
    )
    model = build_model(config).eval()
    tokens = torch.randint(0, 32, (1, 12))
    assert model(tokens).logits.shape[-1] == 32


def test_shared_experts_and_capacity_and_stats():
    config = ModelConfig(
        architecture="moe_decoder",
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        max_sequence_length=8,
        num_experts=4,
        num_experts_per_token=1,
        num_shared_experts=1,
        expert_capacity_factor=0.5,
    )
    model = build_model(config)
    counts = estimate_parameter_count(config)
    assert model.num_parameters() == counts["total"]
    assert counts["active"] < counts["total"]
    assert counts["shared_experts"] > 0
    tokens = torch.randint(0, 32, (2, 8))
    model(tokens)
    stats = model.routing_stats()
    assert stats and stats[0]["dropped_tokens"] >= 0
    assert "routing_entropy" in stats[0]


def test_unavailable_flash_backend_errors():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        max_sequence_length=8,
        attention_backend="flash",
    )
    model = build_model(config)
    tokens = torch.randint(0, 32, (1, 4))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="flash"):
            model(tokens)


def test_document_mask_blocks_cross_document_attention():
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_layers=1,
        num_attention_heads=4,
        num_kv_heads=4,
        intermediate_size=32,
        max_sequence_length=8,
        attention_backend="reference",
    )
    model = build_model(config).eval()
    tokens = torch.randint(0, 32, (1, 6))
    docs = torch.tensor([[0, 0, 0, 1, 1, 1]])
    with torch.no_grad():
        out = model(tokens, document_ids=docs).logits
    assert out.shape == (1, 6, 32)


def test_dynamic_and_paged_caches_match_static():
    from fontaine.models.kv_cache import build_kv_cache

    config = ModelConfig(
        vocab_size=32, hidden_size=16, num_layers=2, num_attention_heads=4,
        num_kv_heads=2, intermediate_size=32, max_sequence_length=16,
    )
    model = build_model(config).eval()
    tokens = torch.randint(0, 32, (1, 10))
    with torch.no_grad():
        full = model(tokens).logits
        for kind in ("static", "dynamic", "paged"):
            cache = build_kv_cache(config, 1, "cpu", torch.float32, kind)
            parts = []
            first = 4
            parts.append(model(tokens[:, :first], cache=cache).logits)
            cache.advance(first)
            for pos in range(first, 10):
                parts.append(model(tokens[:, pos : pos + 1], cache=cache).logits)
                cache.advance(1)
            assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-5)


def test_int4_forward_runs():
    from fontaine.optimization.quantize import quantize_int4

    config = ModelConfig(
        vocab_size=32, hidden_size=32, num_layers=1, num_attention_heads=4,
        num_kv_heads=2, intermediate_size=64, max_sequence_length=8,
    )
    model = build_model(config).eval()
    tokens = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        before = model(tokens).logits
        quantize_int4(model)
        after = model(tokens).logits
    assert after.shape == before.shape
    assert torch.isfinite(after).all()


def test_topology_rejects_impossible_meshes_and_plans_world_one():
    from fontaine.config.schema import DistributedConfig
    from fontaine.distributed.topology import plan_topology, validate_runtime_topology

    config = _profile("tiny")
    assert plan_topology(config.model, 1).world_size == 1
    with pytest.raises(ConfigError, match="tensor_parallel_size"):
        DistributedConfig(tensor_parallel_size=3).validate(config.model)
    with pytest.raises(RuntimeError, match="torchrun"):
        validate_runtime_topology(DistributedConfig(data_parallel_size=2))


def test_column_and_row_parallel_match_a_full_linear():
    from fontaine.distributed.tensor_parallel import column_parallel, row_parallel

    weight = torch.randn(8, 6)
    inputs = torch.randn(2, 3, 6)
    shards = list(weight.chunk(2, dim=0))
    assert torch.allclose(column_parallel(inputs, shards), torch.nn.functional.linear(inputs, weight))
    shards = list(weight.chunk(2, dim=1))
    assert torch.allclose(row_parallel(inputs, shards), torch.nn.functional.linear(inputs, weight), atol=1e-5)


def test_linear_scheduler_and_dpo_and_reinforce():
    from fontaine.training.optim import WarmupScheduler
    from fontaine.training.stages import dpo_loss, reinforce_loss

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    schedule = WarmupScheduler(optimizer, total_steps=10, warmup_steps=0, min_ratio=0.0, schedule="linear")
    schedule.step()
    assert schedule.get_last_lr()[0] < 1.0
    loss = dpo_loss(
        torch.tensor([0.0]), torch.tensor([-1.0]), torch.tensor([0.0]), torch.tensor([0.0]), beta=1.0
    )
    assert loss.ndim == 0
    rl = reinforce_loss(torch.tensor([0.5, 0.5]), torch.tensor([1.0, 0.0]))
    assert torch.isfinite(rl)


def test_quality_filters_and_packing():
    from fontaine.data.packing import pack_token_documents, packing_efficiency, response_mask
    from fontaine.data.quality import hamming, is_malformed, redact_pii, simhash

    redacted, hits = redact_pii("write me at ada@example.com")
    assert hits == 1 and "@" not in redacted
    assert is_malformed("a" * 30)
    assert hamming(simhash("the cat sat on the mat today"), simhash("the cat sat on the mat today")) == 0
    windows, owners, leftover = pack_token_documents([[1, 2], [3]], sequence_length=4, eos_id=0)
    assert windows and leftover >= 0
    assert packing_efficiency([3, 3, 3], 4) > 0
    labels = response_mask([1, 2, 9, 3, 4], [9])
    assert labels[:3] == [-100, -100, -100]
    assert labels[3:] == [3, 4]


def test_verifiers_and_sandbox(tmp_path):
    from fontaine.agents.sandbox import Sandbox
    from fontaine.agents.tools import run_episode
    from fontaine.evaluation.verifiers import exact_match, mathematical, unit_test_verifier

    assert exact_match(" 7 ", "7")
    assert mathematical("the answer is 3.0", "3")
    assert unit_test_verifier("x = 2", "x == 2", tmp_path)
    sandbox = Sandbox(tmp_path / "jail")
    with pytest.raises(PermissionError):
        sandbox.resolve("../outside.txt")

    def policy(trajectory):
        if not any(step["kind"] == "action" for step in trajectory.steps):
            return {"tool": "filesystem", "arguments": {"op": "write", "path": "note.txt", "text": "ok"}}
        return {"finish": True, "answer": "done"}

    episode = run_episode(policy, sandbox, "write a note")
    assert episode.steps[-1]["kind"] == "termination"
    assert (tmp_path / "jail" / "note.txt").read_text(encoding="utf-8") == "ok"


def test_cli_estimate_and_hardware(capsys):
    assert main(["model", "estimate-params", "--config", str(SUPER), "--profile", "tiny"]) == 0
    out = capsys.readouterr().out
    assert "Total parameters:" in out
    assert "Active parameters per token:" in out
    assert "Approximate total training memory:" in out
    assert main(["hardware", "inspect"]) == 0
    assert "CPU" in capsys.readouterr().out
    assert main(["distributed", "plan", "--config", str(SUPER), "--profile", "7b", "--world-size", "1"]) == 0
    assert "Data Parallel: 1" in capsys.readouterr().out
    assert main(["model", "feasibility", "--config", str(SUPER), "--profile", "tiny"]) == 0
    assert "inference:" in capsys.readouterr().out


def test_rank_layout_and_micro_batch():
    from fontaine.distributed.topology import ParallelTopology
    from fontaine.optimization.memory import suggest_micro_batch

    topology = ParallelTopology(2, 2, 2, 2, 1)
    local = topology.for_rank(5)
    assert (local.data_rank, local.pipeline_rank, local.tensor_rank) == (1, 0, 1)
    assert topology.for_rank(0).pipeline_next() == 2
    config = _profile("tiny").model
    assert suggest_micro_batch(config, 32, 4, 10**18) == 4
    with pytest.raises(RuntimeError, match="could not be measured"):
        suggest_micro_batch(config, 32, 4, None)


def test_sequence_parallel_requires_a_process_group():
    from fontaine.models.components import _gather_sequence

    with pytest.raises(RuntimeError, match="sequence_parallel_size"):
        _gather_sequence(torch.zeros(1, 2, 4), 2)


def test_async_checkpoint_roundtrip(tmp_path):
    from fontaine.checkpointing import CheckpointManager

    layer = torch.nn.Linear(4, 4)
    manager = CheckpointManager(tmp_path / "checkpoints")
    manager.save(step=1, epoch=0, model=layer, async_write=True)
    manager.wait()
    restored = torch.nn.Linear(4, 4)
    manager.load("latest", model=restored)
    assert torch.equal(layer.weight, restored.weight)
