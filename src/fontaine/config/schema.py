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
VALID_SCHEDULERS = ("cosine", "constant")
VALID_TOKENIZER_TYPES = ("char", "hf_bpe")


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
    # Display label for ``fontaine model inspect`` (empty for the dense ladder).
    cpu_tier: str = ""

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
        if self.hidden_size % self.num_attention_heads != 0:
            raise ConfigError(
                f"model.hidden_size ({self.hidden_size}) must be divisible by "
                f"model.num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_kv_heads < 1 or self.num_attention_heads % self.num_kv_heads != 0:
            raise ConfigError(
                f"model.num_attention_heads ({self.num_attention_heads}) must be a "
                f"multiple of model.num_kv_heads ({self.num_kv_heads})"
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

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


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


@dataclass
class FontaineConfig:
    """Top-level configuration assembled from per-subsystem YAML sections."""

    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def validate(self) -> None:
        self.model.validate()
        self.tokenizer.validate()
        self.data.validate()
        self.training.validate()
        self.evaluation.validate()
        self.inference.validate()
        if self.data.sequence_length > self.model.max_sequence_length:
            raise ConfigError(
                f"data.sequence_length ({self.data.sequence_length}) must be <= "
                f"model.max_sequence_length ({self.model.max_sequence_length})"
            )
