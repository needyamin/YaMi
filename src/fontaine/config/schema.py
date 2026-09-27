"""Typed configuration schemas.

Every configurable knob of Fontaine lives here as a dataclass field with a
default. The defaults describe the *tiny* development model so that an empty
config is valid; real runs override values through YAML files.

Design rules:
- Dataclasses, not hand-parsed dicts: validation is centralized and typos in
  YAML keys fail loudly (see ``fontaine.config.loader``).
- No field may reference another subsystem: configs are plain data, so they
  can be serialized into checkpoints, manifests, and experiment directories.
"""

from dataclasses import dataclass, field
from typing import Any

from fontaine.config.errors import ConfigError

VALID_ARCHITECTURES = ("decoder_transformer", "moe_decoder")
VALID_ACTIVATIONS = ("swiglu", "gelu")
VALID_NORMS = ("rmsnorm", "layernorm")
VALID_PRECISIONS = ("auto", "fp32", "bf16", "fp16")
VALID_ROPE_SCALING = ("none", "linear", "ntk")
VALID_INFERENCE_PRECISIONS = ("auto", "fp32", "bf16", "fp16", "int8", "int4")
VALID_SCHEDULERS = ("cosine", "linear", "constant")
VALID_TOKENIZER_TYPES = ("char", "hf_bpe")
VALID_ATTENTION_TYPES = ("auto", "standard", "gqa", "sliding_window", "sparse", "hybrid")
VALID_ATTENTION_BACKENDS = ("auto", "optimized", "reference", "flash")
VALID_KERNELS = ("auto", "optimized", "reference")
VALID_LAYER_KINDS = ("full", "sliding", "sparse")
VALID_SPARSE_PATTERNS = ("local", "block", "strided", "global-local", "hybrid")
VALID_ROUTERS = ("top_k",)
VALID_STAGES = (
    "pretraining",
    "supervised_fine_tuning",
    "preference_training",
    "reinforcement_learning",
)
VALID_SHARDING = ("none", "optimizer", "fsdp")
VALID_OPTIMIZERS = ("adamw",)
VALID_KV_CACHES = ("static", "dynamic", "paged")
VALID_SCRIPT_FILTERS = ("", "latin", "cyrillic", "cjk", "arabic", "devanagari")
IMPLEMENTED_MODALITIES = ("text",)


@dataclass
class ModelConfig:
    """Architecture of the decoder-only Transformer.

    These fields map 1:1 to the architecture knobs described in
    ``docs/model/design.md``. The same schema covers Fontaine Tiny through
    Fontaine XL -- only the values change, never the code.
    """

    # ``decoder_transformer`` is dense. ``moe_decoder`` swaps the feed-forward
    # for a mixture of experts (see fontaine.models.factory).
    architecture: str = "decoder_transformer"
    # int, or the literal string "auto" (= resolve from the trained tokenizer
    # at train time; see fontaine.tokenizer.registry.resolve_vocab_size).
    vocab_size: int | str = 512
    hidden_size: int = 256
    num_layers: int = 4
    num_attention_heads: int = 4
    num_kv_heads: int = 4  # < num_attention_heads enables grouped-query attention
    intermediate_size: int = 768
    max_sequence_length: int = 256
    rope_theta: float = 10000.0
    dropout: float = 0.0
    activation: str = "swiglu"
    normalization: str = "rmsnorm"
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    # Mixture-of-experts feed-forward. ``num_experts == 1`` keeps the dense MLP
    # (no router). ``moe_decoder`` requires at least two experts.
    num_experts: int = 1
    num_experts_per_token: int = 1
    moe_aux_loss_coef: float = 0.01
    # RMSNorm on each query and key head before RoPE (Qwen3, Gemma 3, OLMo 2).
    qk_norm: bool = False
    # Local attention over the last ``sliding_window`` tokens (0 = full attention).
    # With ``global_attention_every = k``, every k-th layer keeps full attention;
    # 0 makes every layer local.
    sliding_window: int = 0
    global_attention_every: int = 0
    # Stretch RoPE past the trained length: ``linear`` divides positions by the
    # factor, ``ntk`` raises the wavelength base. Set max_sequence_length to the
    # extended window.
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    # Display label for ``fontaine model inspect`` (empty for the dense ladder).
    cpu_tier: str = ""
    # 0 keeps the historical rule ``head_dim = hidden_size // num_attention_heads``.
    # A positive value is used as-is and does not have to multiply back to hidden_size.
    # YAML key ``head_dim`` is accepted and stored here (see the config loader).
    explicit_head_dim: int = 0
    attention_type: str = "auto"
    attention_backend: str = "auto"
    kernel: str = "auto"
    layer_pattern: list[str] = field(default_factory=list)
    sparse_pattern: str = "local"
    sparse_block_size: int = 64
    sparse_stride: int = 16
    sparse_window: int = 0
    sparse_global_tokens: int = 1
    num_shared_experts: int = 0
    # 0 means routed experts use ``intermediate_size``.
    expert_intermediate_size: int = 0
    # 0 disables token dropping. A positive factor caps tokens per expert.
    expert_capacity_factor: float = 0.0
    load_balancing_enabled: bool = True
    router_type: str = "top_k"
    # 0 means unset. When scaling is enabled it must agree with the factor.
    rope_original_context_length: int = 0
    architecture_version: str = "1"
    modalities: list[str] = field(default_factory=lambda: ["text"])

    def validate(self) -> None:
        if self.architecture not in VALID_ARCHITECTURES:
            raise ConfigError(f"model.architecture must be one of {VALID_ARCHITECTURES}")
        if isinstance(self.vocab_size, str):
            if self.vocab_size != "auto":
                raise ConfigError('model.vocab_size must be an integer or the literal "auto"')
        elif self.vocab_size < 1:
            raise ConfigError("model.vocab_size must be >= 1")
        if self.hidden_size < 1 or self.num_layers < 1 or self.num_attention_heads < 1:
            raise ConfigError("model.hidden_size/num_layers/num_attention_heads must be >= 1")
        if self.explicit_head_dim == 0:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ConfigError(
                    f"model.hidden_size ({self.hidden_size}) must be divisible by "
                    f"model.num_attention_heads ({self.num_attention_heads}). "
                    f"Set model.head_dim to an explicit even size if the projection "
                    f"width should differ from hidden_size."
                )
        elif self.explicit_head_dim < 1 or self.explicit_head_dim % 2 != 0:
            raise ConfigError(
                f"model.head_dim={self.explicit_head_dim} must be a positive even integer "
                f"(RoPE rotates pairs of dimensions)."
            )
        if self.num_kv_heads < 1 or self.num_attention_heads % self.num_kv_heads != 0:
            raise ConfigError(
                f"num_attention_heads={self.num_attention_heads} is not divisible by "
                f"num_kv_heads={self.num_kv_heads}. "
                f"Choose a num_kv_heads value that divides {self.num_attention_heads}."
            )
        if self.intermediate_size < 1:
            raise ConfigError("model.intermediate_size must be >= 1")
        if self.max_sequence_length < 2:
            raise ConfigError("model.max_sequence_length must be >= 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        if self.activation not in VALID_ACTIVATIONS:
            raise ConfigError(f"model.activation must be one of {VALID_ACTIVATIONS}")
        if self.normalization not in VALID_NORMS:
            raise ConfigError(f"model.normalization must be one of {VALID_NORMS}")
        if self.norm_eps <= 0:
            raise ConfigError("model.norm_eps must be > 0")
        if self.num_experts < 1:
            raise ConfigError("model.num_experts must be >= 1")
        if not 1 <= self.num_experts_per_token <= self.num_experts:
            raise ConfigError(
                "model.num_experts_per_token must be between 1 and model.num_experts"
            )
        if self.moe_aux_loss_coef < 0:
            raise ConfigError("model.moe_aux_loss_coef must be >= 0")
        if self.architecture == "decoder_transformer" and self.num_experts != 1:
            raise ConfigError(
                "decoder_transformer requires model.num_experts=1; "
                "use architecture moe_decoder for a mixture of experts"
            )
        if self.architecture == "moe_decoder" and self.num_experts < 2:
            raise ConfigError("moe_decoder requires model.num_experts >= 2")
        if self.sliding_window < 0:
            raise ConfigError("model.sliding_window must be >= 0 (0 = full attention)")
        if self.global_attention_every < 0:
            raise ConfigError("model.global_attention_every must be >= 0")
        if self.rope_scaling_type not in VALID_ROPE_SCALING:
            raise ConfigError(f"model.rope_scaling_type must be one of {VALID_ROPE_SCALING}")
        if self.rope_scaling_factor < 1.0:
            raise ConfigError("model.rope_scaling_factor must be >= 1.0")
        self._validate_attention()
        self._validate_moe_extensions()
        self._validate_rope_original_context()
        if not self.architecture_version:
            raise ConfigError("model.architecture_version must be a non-empty string")
        if not self.modalities:
            raise ConfigError("model.modalities must list at least one modality")
        for modality in self.modalities:
            if modality not in IMPLEMENTED_MODALITIES:
                raise ConfigError(
                    f"modality {modality!r} is not implemented. "
                    f"YaMi currently runs text only "
                    f"(implemented: {IMPLEMENTED_MODALITIES})."
                )

    def _validate_attention(self) -> None:
        if self.attention_type not in VALID_ATTENTION_TYPES:
            raise ConfigError(f"model.attention_type must be one of {VALID_ATTENTION_TYPES}")
        if self.attention_backend not in VALID_ATTENTION_BACKENDS:
            raise ConfigError(
                f"model.attention_backend must be one of {VALID_ATTENTION_BACKENDS}"
            )
        if self.kernel not in VALID_KERNELS:
            raise ConfigError(f"model.kernel must be one of {VALID_KERNELS}")
        if self.sparse_pattern not in VALID_SPARSE_PATTERNS:
            raise ConfigError(f"model.sparse_pattern must be one of {VALID_SPARSE_PATTERNS}")
        if self.sparse_block_size < 1:
            raise ConfigError("model.sparse_block_size must be >= 1")
        if self.sparse_stride < 1:
            raise ConfigError("model.sparse_stride must be >= 1")
        if self.sparse_window < 0 or self.sparse_global_tokens < 0:
            raise ConfigError("model.sparse_window and sparse_global_tokens must be >= 0")
        for name in self.layer_pattern:
            if name not in VALID_LAYER_KINDS:
                raise ConfigError(
                    f"model.layer_pattern entries must be one of {VALID_LAYER_KINDS} "
                    f"(got {name!r})"
                )
        if self.layer_pattern and self.global_attention_every:
            raise ConfigError(
                "model.layer_pattern and model.global_attention_every are both set. "
                "Use layer_pattern to choose full, sliding, and sparse layers, "
                "or global_attention_every with sliding_window, not both."
            )
        if self.attention_type == "standard" and self.num_kv_heads != self.num_attention_heads:
            raise ConfigError(
                f"attention type standard requires num_kv_heads == num_attention_heads "
                f"({self.num_kv_heads} != {self.num_attention_heads}). "
                f"Use attention type gqa for grouped-query attention."
            )
        if self.attention_type == "sliding_window" and self.sliding_window < 1:
            raise ConfigError(
                "attention type sliding_window requires model.sliding_window >= 1"
            )
        if self.attention_type == "hybrid" and not self.layer_pattern:
            raise ConfigError(
                "attention type hybrid requires model.layer_pattern "
                "(for example [full, sliding, sliding, sparse])"
            )
        if self.attention_type == "sparse" and self.layer_pattern:
            if any(name != "sparse" for name in self.layer_pattern):
                raise ConfigError(
                    "attention type sparse only allows layer_pattern entries of 'sparse', "
                    "or leave layer_pattern empty to make every layer sparse"
                )
        needs_span = self.attention_type == "sparse" or "sparse" in self.layer_pattern
        span = self.sparse_window or self.sliding_window
        if needs_span and self.sparse_pattern in ("local", "strided", "global-local", "hybrid"):
            if span < 1:
                raise ConfigError(
                    f"sparse pattern {self.sparse_pattern!r} needs a positive window. "
                    f"Set model.sparse_window or model.sliding_window."
                )
        if any(name == "sliding" for name in self.layer_pattern) and self.sliding_window < 1:
            raise ConfigError(
                "layer_pattern contains 'sliding' but model.sliding_window is 0. "
                "Set sliding_window to the local attention span."
            )

    def _validate_moe_extensions(self) -> None:
        if self.router_type not in VALID_ROUTERS:
            raise ConfigError(
                f"model.router_type must be one of {VALID_ROUTERS}. "
                f"Other routers are not implemented."
            )
        if self.num_shared_experts < 0:
            raise ConfigError("model.num_shared_experts must be >= 0")
        if self.expert_intermediate_size < 0:
            raise ConfigError("model.expert_intermediate_size must be >= 0")
        if self.expert_capacity_factor < 0:
            raise ConfigError("model.expert_capacity_factor must be >= 0 (0 disables dropping)")
        if self.architecture == "decoder_transformer" and self.num_shared_experts:
            raise ConfigError(
                "decoder_transformer has no experts; num_shared_experts must be 0. "
                "Use architecture moe_decoder to add shared experts."
            )
        if self.architecture == "decoder_transformer" and self.expert_intermediate_size:
            raise ConfigError(
                "decoder_transformer uses intermediate_size for its feed-forward. "
                "expert_intermediate_size applies only to moe_decoder."
            )
        if self.architecture == "decoder_transformer" and self.expert_capacity_factor:
            raise ConfigError(
                "expert_capacity_factor applies only to moe_decoder "
                "(decoder_transformer has a single feed-forward)"
            )

    def _validate_rope_original_context(self) -> None:
        original = self.rope_original_context_length
        if original < 0:
            raise ConfigError("model.rope_original_context_length must be >= 0")
        if original == 0:
            return
        if self.rope_scaling_type == "none":
            if original != self.max_sequence_length:
                raise ConfigError(
                    f"rope_scaling_type is none, so rope_original_context_length "
                    f"({original}) must be 0 or equal max_sequence_length "
                    f"({self.max_sequence_length})."
                )
            return
        if original > self.max_sequence_length:
            raise ConfigError(
                f"rope_original_context_length ({original}) cannot exceed "
                f"max_sequence_length ({self.max_sequence_length})"
            )
        expected = self.max_sequence_length / original
        if abs(self.rope_scaling_factor - expected) > 1e-4:
            raise ConfigError(
                f"rope_scaling_factor={self.rope_scaling_factor} does not match "
                f"max_sequence_length/rope_original_context_length "
                f"({self.max_sequence_length}/{original} = {expected:.6g}). "
                f"Set the factor to that ratio, or leave rope_original_context_length at 0."
            )

    @property
    def head_dim(self) -> int:
        if self.explicit_head_dim > 0:
            return self.explicit_head_dim
        return self.hidden_size // self.num_attention_heads

    def resolved_expert_intermediate(self) -> int:
        """Width of one routed or shared expert. Dense models use ``intermediate_size``."""
        if self.expert_intermediate_size > 0:
            return self.expert_intermediate_size
        return self.intermediate_size

    def layer_kind(self, layer_idx: int) -> str:
        """``full``, ``sliding``, or ``sparse`` for one block."""
        if self.layer_pattern:
            return self.layer_pattern[layer_idx % len(self.layer_pattern)]
        if self.attention_type == "sparse":
            return "sparse"
        if self.layer_window(layer_idx) > 0:
            return "sliding"
        return "full"

    def layer_window(self, layer_idx: int) -> int:
        """Attention window for one layer; 0 means full causal attention."""
        if self.sliding_window <= 0:
            return 0
        if self.global_attention_every > 0 and (layer_idx + 1) % self.global_attention_every == 0:
            return 0
        return self.sliding_window


@dataclass
class TokenizerConfig:
    """Tokenizer subsystem settings (independent from the model)."""

    type: str = "char"
    # Where a trained tokenizer is stored / loaded from.
    path: str | None = None
    lowercase: bool = False
    # Byte-level BPE hyperparameters (used only when type == "hf_bpe").
    vocab_size: int = 4096
    min_frequency: int = 2
    # Extra user-defined special tokens beyond the built-ins.
    special_tokens: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.type not in VALID_TOKENIZER_TYPES:
            raise ConfigError(f"tokenizer.type must be one of {VALID_TOKENIZER_TYPES}")
        if self.vocab_size < 256 and self.type == "hf_bpe":
            raise ConfigError("tokenizer.vocab_size must be >= 256 for byte-level BPE")
        for token in self.special_tokens:
            if not token or " " in token:
                raise ConfigError(
                    f"tokenizer.special_tokens entries must be non-empty without spaces "
                    f"(got {token!r})"
                )


@dataclass
class DataConfig:
    """Dataset preparation and loading settings.

    ``raw_paths`` are inputs for the preparation pipeline; ``manifest_path``
    points at a prepared dataset (shards + manifest) consumed by training.
    """

    raw_paths: list[str] = field(default_factory=list)
    output_dir: str = "datasets/prepared"
    manifest_path: str | None = None
    val_fraction: float = 0.05
    # Cleaning / filtering (streaming, bounded memory).
    normalize_unicode: bool = True
    strip_whitespace: bool = True
    min_document_chars: int = 32
    max_document_chars: int = 1_000_000
    drop_duplicates: bool = True
    # Packing / sharding.
    sequence_length: int = 256
    shard_size_tokens: int = 5_000_000
    dtype: str = "auto"  # "auto" | "uint16" | "uint32"
    seed: int = 42
    # Each entry is {name, weight, path}. Empty keeps the single raw_paths corpus.
    mixture: list[dict[str, Any]] = field(default_factory=list)
    # When true, packed windows carry document ids so attention cannot cross documents.
    pack_documents: bool = False
    near_dedup: bool = False
    # Hamming distance on 64-bit simhash. Used only when near_dedup is true.
    near_dedup_max_distance: int = 3
    drop_pii: bool = False
    drop_malformed: bool = False
    # Empty disables. This is a Unicode-script heuristic, not a language-id model.
    script_filter: str = ""
    contamination_paths: list[str] = field(default_factory=list)
    # Fraction of 5-word shingles that must hit the contamination set to drop a document.
    contamination_shingle_fraction: float = 0.8

    def validate(self) -> None:
        # NOTE: requiring raw_paths/manifest_path is enforced by the data
        # commands only — inference/training-adjacent sections may carry an
        # unused default DataConfig.
        if not 0.0 <= self.val_fraction < 1.0:
            raise ConfigError("data.val_fraction must be in [0, 1)")
        if self.min_document_chars < 1 or self.max_document_chars < self.min_document_chars:
            raise ConfigError(
                "data.min_document_chars/max_document_chars are inconsistent "
                "(need 1 <= min <= max)"
            )
        if self.sequence_length < 2:
            raise ConfigError("data.sequence_length must be >= 2")
        if self.shard_size_tokens < self.sequence_length:
            raise ConfigError("data.shard_size_tokens must be >= data.sequence_length")
        if self.dtype not in ("auto", "uint16", "uint32"):
            raise ConfigError('data.dtype must be "auto", "uint16", or "uint32"')
        if self.mixture and self.raw_paths:
            raise ConfigError(
                "set data.mixture or data.raw_paths, not both. "
                "A mixture already names each source path."
            )
        for entry in self.mixture:
            if not isinstance(entry, dict):
                raise ConfigError("data.mixture entries must be mappings")
            missing = {"name", "weight", "path"} - set(entry)
            if missing:
                raise ConfigError(
                    f"data.mixture entry is missing {sorted(missing)} "
                    f"(need name, weight, and path)"
                )
            if float(entry["weight"]) <= 0:
                raise ConfigError(
                    f"data.mixture weight for {entry.get('name')!r} must be > 0"
                )
        if self.near_dedup_max_distance < 0:
            raise ConfigError("data.near_dedup_max_distance must be >= 0")
        if self.script_filter not in VALID_SCRIPT_FILTERS:
            raise ConfigError(f"data.script_filter must be one of {VALID_SCRIPT_FILTERS}")
        if not 0.0 < self.contamination_shingle_fraction <= 1.0:
            raise ConfigError("data.contamination_shingle_fraction must be in (0, 1]")
        if self.contamination_paths and self.contamination_shingle_fraction <= 0:
            raise ConfigError("contamination filtering needs a positive shingle fraction")


@dataclass
class TrainingConfig:
    """Training-engine settings.

    The trainer contains no dataset- or model-specific logic; everything
    variable lives here. See ``docs/training/process.md``.
    """

    run_name: str = "fontaine-run"
    experiments_dir: str = "experiments"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    max_steps: int = 1000
    batch_size: int = 8  # micro-batch (per step, before accumulation)
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    min_learning_rate_ratio: float = 0.1  # final LR = learning_rate * ratio
    lr_scheduler: str = "cosine"
    warmup_steps: int = 50
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0
    precision: str = "auto"
    gradient_checkpointing: bool = False
    eval_interval: int = 100
    log_interval: int = 10
    checkpoint_interval: int = 200
    keep_last_n_checkpoints: int = 2
    early_stopping_patience: int = 0  # 0 disables early stopping
    seed: int = 42
    deterministic: bool = False
    stage: str = "pretraining"
    optimizer: str = "adamw"
    adam_eps: float = 1e-8
    # none: replicate. optimizer: shard AdamW state (ZeRO-1). fsdp: shard parameters,
    # gradients, and optimizer state. Values other than none require torch.distributed.
    sharding: str = "none"
    # When true, lower batch_size until the memory estimate fits detected RAM. Never raises it.
    fit_batch_to_memory: bool = False
    async_checkpoints: bool = False

    def validate(self) -> None:
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ConfigError('training.device must be one of "auto", "cpu", "cuda", "mps"')
        if self.max_steps < 1:
            raise ConfigError("training.max_steps must be >= 1")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ConfigError("training.batch_size / gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ConfigError("training.learning_rate must be > 0")
        if not 0.0 <= self.min_learning_rate_ratio <= 1.0:
            raise ConfigError("training.min_learning_rate_ratio must be in [0, 1]")
        if self.lr_scheduler not in VALID_SCHEDULERS:
            raise ConfigError(f"training.lr_scheduler must be one of {VALID_SCHEDULERS}")
        if self.stage not in VALID_STAGES:
            raise ConfigError(f"training.stage must be one of {VALID_STAGES}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ConfigError(
                f"training.optimizer must be one of {VALID_OPTIMIZERS}. "
                f"Other optimizers are not implemented."
            )
        if self.adam_eps <= 0:
            raise ConfigError("training.adam_eps must be > 0")
        if self.sharding not in VALID_SHARDING:
            raise ConfigError(f"training.sharding must be one of {VALID_SHARDING}")
        if self.warmup_steps < 0:
            raise ConfigError("training.warmup_steps must be >= 0")
        if self.precision not in VALID_PRECISIONS:
            raise ConfigError(f"training.precision must be one of {VALID_PRECISIONS}")
        if self.log_interval < 1 or self.eval_interval < 1 or self.checkpoint_interval < 1:
            raise ConfigError("training intervals (log/eval/checkpoint) must be >= 1")
        if self.keep_last_n_checkpoints < 1:
            raise ConfigError("training.keep_last_n_checkpoints must be >= 1")
        if self.early_stopping_patience < 0:
            raise ConfigError("training.early_stopping_patience must be >= 0")


@dataclass
class EvaluationConfig:
    """Evaluation subsystem settings.

    ``evaluators`` is a list of ``{name, params}`` entries resolved through the
    evaluator registry, so new benchmarks are added via configuration (and a
    registered class) without touching the training engine.
    """

    evaluators: list[dict[str, Any]] = field(
        default_factory=lambda: [{"name": "validation_loss", "params": {"max_batches": 64}}]
    )
    split: str = "val"

    def validate(self) -> None:
        if self.split not in ("train", "val"):
            raise ConfigError('evaluation.split must be "train" or "val"')
        for entry in self.evaluators:
            if not isinstance(entry, dict) or "name" not in entry:
                raise ConfigError(
                    "evaluation.evaluators entries must be objects with a 'name' field"
                )


@dataclass
class InferenceConfig:
    """Generation / serving settings (independent from training)."""

    temperature: float = 0.8
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    repetition_penalty: float = 1.0  # 1.0 disables
    max_new_tokens: int = 256
    stop_sequences: list[str] = field(default_factory=list)
    seed: int | None = None
    # Name shown in chat model pickers (Ollama /api/tags, /api/ps, /api/show).
    model_name: str = "Yami v1.0"
    # Weight format at inference: ``auto`` picks int8 on CPU for models of 20M
    # parameters or more, bf16 on a GPU that supports it, fp32 otherwise.
    precision: str = "auto"
    # CPU threads for generation (0 = PyTorch's physical-core default).
    num_threads: int = 0
    # static preallocates the full window. dynamic and paged grow with the sequence.
    kv_cache: str = "static"
    # 0 disables speculative decoding. A positive count requires draft_checkpoint.
    speculative_tokens: int = 0
    draft_checkpoint: str = ""
    # Dev-only HTTP server.
    server_host: str = "127.0.0.1"
    server_port: int = 8321

    def validate(self) -> None:
        if self.temperature < 0:
            raise ConfigError("inference.temperature must be >= 0 (0 = greedy)")
        if self.top_k < 0:
            raise ConfigError("inference.top_k must be >= 0 (0 = disabled)")
        if not 0.0 < self.top_p <= 1.0:
            raise ConfigError("inference.top_p must be in (0, 1] (1.0 = disabled)")
        if self.repetition_penalty < 1.0:
            raise ConfigError("inference.repetition_penalty must be >= 1.0 (1.0 = disabled)")
        if self.max_new_tokens < 1:
            raise ConfigError("inference.max_new_tokens must be >= 1")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ConfigError("inference.model_name must be a non-empty string")
        if self.precision not in VALID_INFERENCE_PRECISIONS:
            raise ConfigError(f"inference.precision must be one of {VALID_INFERENCE_PRECISIONS}")
        if self.num_threads < 0:
            raise ConfigError("inference.num_threads must be >= 0 (0 = automatic)")
        if self.kv_cache not in VALID_KV_CACHES:
            raise ConfigError(f"inference.kv_cache must be one of {VALID_KV_CACHES}")
        if self.speculative_tokens < 0:
            raise ConfigError("inference.speculative_tokens must be >= 0")
        if self.speculative_tokens > 0 and not self.draft_checkpoint:
            raise ConfigError(
                "inference.speculative_tokens is "
                f"{self.speculative_tokens} but inference.draft_checkpoint is empty. "
                "Point draft_checkpoint at a checkpoint, or set speculative_tokens to 0."
            )
        if self.draft_checkpoint and self.speculative_tokens == 0:
            raise ConfigError(
                "inference.draft_checkpoint is set but speculative_tokens is 0. "
                "Set speculative_tokens to the number of draft tokens per step, "
                "or clear draft_checkpoint."
            )


@dataclass
class DistributedConfig:
    """Parallel sizes. All ones is the single-process trainer.

    ``data * tensor * pipeline`` is the process count. Expert parallel must
    divide data parallel, because experts are sharded inside each data-parallel
    group. Sequence parallel is 1 or equal to tensor parallel.
    """

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel_size: int = 1
    expert_parallel_size: int = 1

    @property
    def world_size(self) -> int:
        return (
            self.data_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
        )

    def validate(self, model: ModelConfig) -> None:
        sizes = {
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "sequence_parallel_size": self.sequence_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
        }
        for name, value in sizes.items():
            if value < 1:
                raise ConfigError(f"distributed.{name} must be >= 1 (got {value})")
        tp = self.tensor_parallel_size
        if self.sequence_parallel_size not in (1, tp):
            raise ConfigError(
                f"distributed.sequence_parallel_size={self.sequence_parallel_size} "
                f"must be 1 or equal tensor_parallel_size ({tp})."
            )
        if model.num_attention_heads % tp != 0 or model.num_kv_heads % tp != 0:
            raise ConfigError(
                f"tensor_parallel_size={tp} must divide num_attention_heads="
                f"{model.num_attention_heads} and num_kv_heads={model.num_kv_heads}."
            )
        if model.intermediate_size % tp != 0:
            raise ConfigError(
                f"tensor_parallel_size={tp} must divide intermediate_size="
                f"{model.intermediate_size}."
            )
        expert_width = model.resolved_expert_intermediate()
        if model.num_experts > 1 and expert_width % tp != 0:
            raise ConfigError(
                f"tensor_parallel_size={tp} must divide expert intermediate size "
                f"{expert_width}."
            )
        pp = self.pipeline_parallel_size
        if pp > model.num_layers:
            raise ConfigError(
                f"pipeline_parallel_size={pp} is larger than num_layers="
                f"{model.num_layers}. A stage needs at least one layer."
            )
        ep = self.expert_parallel_size
        if ep > 1 and model.num_experts <= 1:
            raise ConfigError(
                "expert_parallel_size > 1 requires architecture moe_decoder "
                "with num_experts >= 2."
            )
        if ep > 1 and model.num_experts % ep != 0:
            raise ConfigError(
                f"expert_parallel_size={ep} must divide num_experts={model.num_experts}."
            )
        if ep > 1 and self.data_parallel_size % ep != 0:
            raise ConfigError(
                f"expert_parallel_size={ep} must divide data_parallel_size="
                f"{self.data_parallel_size}. Experts are sharded inside the "
                f"data-parallel group, so world_size stays "
                f"data_parallel * tensor_parallel * pipeline_parallel "
                f"(= {self.world_size} with the current sizes)."
            )


@dataclass
class FontaineConfig:
    """Top-level configuration assembled from per-subsystem YAML sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    def validate(self) -> None:
        self.model.validate()
        self.tokenizer.validate()
        self.data.validate()
        self.training.validate()
        self.evaluation.validate()
        self.inference.validate()
        self.distributed.validate(self.model)
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ConfigError(
                f"data.sequence_length ({self.data.sequence_length}) must be <= "
                f"model.max_sequence_length ({self.model.max_sequence_length})"
            )
        if (
            self.distributed.sequence_parallel_size > 1
            and self.data.sequence_length % self.distributed.sequence_parallel_size != 0
        ):
            raise ConfigError(
                f"data.sequence_length ({self.data.sequence_length}) must be divisible "
                f"by sequence_parallel_size ({self.distributed.sequence_parallel_size})."
            )
