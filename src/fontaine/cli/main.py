"""Fontaine command-line interface.

Commands (all accept ``--config file.yaml`` repeatably plus ``--set k=v``
overrides for anything in the config tree):

    fontaine data validate      verify raw inputs and/or a prepared manifest
    fontaine data prepare       raw data -> cleaned/packed token shards + manifest
    fontaine tokenizer train    train a tokenizer on the configured corpus
    fontaine tokenizer test     encode/decode round-trip smoke test
    fontaine model inspect      parameter counts + memory estimates
    fontaine train              end-to-end training (prepares data if needed)
    fontaine evaluate           run configured evaluators against a checkpoint
    fontaine generate           stream text generation from a checkpoint
    fontaine checkpoint inspect show checkpoint metadata
    fontaine checkpoint convert export weights (e.g. to safetensors)
    fontaine serve              start the local inference HTTP API
"""

import argparse
import json
import sys
from pathlib import Path

from fontaine import __version__
from fontaine.config.errors import ConfigError
from fontaine.config.loader import load_fontaine_config
from fontaine.config.schema import FontaineConfig
from fontaine.utils.logging import configure_logging, get_logger

logger = get_logger("cli")

_CONFIG_FLAGS = {
    "--model-config": None,
    "--training-config": None,
    "--data-config": None,
    "--inference-config": None,
}


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="FILE",
        help="config YAML (repeatable; later files win)",
    )
    parser.add_argument("--model-config", help="shorthand: config file providing the model section")
    parser.add_argument("--training-config", help="shorthand: config file providing the training section")
    parser.add_argument("--data-config", help="shorthand: config file providing the data section")
    parser.add_argument("--inference-config", help="shorthand: config file providing the inference section")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any config value, e.g. --set training.max_steps=10",
    )
    parser.add_argument("--profile", default=None, help="named profile defined in the config file")


def _load(args: argparse.Namespace) -> FontaineConfig:
    files = list(args.config)
    for flag in ("model_config", "training_config", "data_config", "inference_config"):
        value = getattr(args, flag)
        if value:
            files.append(value)
    return load_fontaine_config(files, overrides=args.set, profile=getattr(args, "profile", None))


# -- data --------------------------------------------------------------------


def cmd_data_validate(args: argparse.Namespace) -> int:
    config = _load(args)
    if not config.data.manifest_path and not config.data.raw_paths:
        raise ConfigError("nothing to validate: set data.manifest_path or data.raw_paths")
    if config.data.manifest_path:
        from fontaine.data.manifest import DatasetManifest

        directory = Path(config.data.manifest_path).parent
        manifest = DatasetManifest.load(directory)
        manifest.validate(directory, verify_checksums=not args.skip_checksums)
        logger.info(
            "manifest OK: %s v%s | documents=%d | train_tokens=%d | val_tokens=%d",
            manifest.name,
            manifest.version,
            manifest.num_documents,
            manifest.tokens_per_split.get("train", 0),
            manifest.tokens_per_split.get("val", 0),
        )
    if config.data.raw_paths:
        from fontaine.data.sources import iter_source_documents

        count = 0
        total_chars = 0
        for document in iter_source_documents(config.data.raw_paths):
            count += 1
            total_chars += len(document)
        if count == 0:
            raise ConfigError("raw data paths yielded zero documents")
        logger.info(
            "raw data OK: %d documents, %d chars (avg %.1f)",
            count,
            total_chars,
            total_chars / count,
        )
    if not config.data.manifest_path and not config.data.raw_paths:
        raise ConfigError("nothing to validate: set data.manifest_path or data.raw_paths")
    return 0


def cmd_data_prepare(args: argparse.Namespace) -> int:
    from fontaine.data import prepare_dataset
    from fontaine.tokenizer.registry import load_tokenizer

    config = _load(args)
    if not config.data.raw_paths:
        raise ConfigError("data.raw_paths must list at least one input file/directory")
    tokenizer = load_tokenizer(args.tokenizer_dir)
    result = prepare_dataset(
        config.data,
        tokenizer,
        name=args.name,
        dataset_version=args.version,
        license=args.license,
        language=args.language,
        notes=args.notes,
    )
    print(json.dumps(result.summary(), indent=2))
    return 0


# -- tokenizer -----------------------------------------------------------------


def cmd_tokenizer_train(args: argparse.Namespace) -> int:
    from fontaine.data.cleaning import DocumentCleaner
    from fontaine.data.sources import iter_source_documents
    from fontaine.tokenizer.registry import train_tokenizer

    config = _load(args)
    cleaner = DocumentCleaner(config.data)
    if not config.data.raw_paths:
        raise ConfigError("data.raw_paths must list at least one input file/directory")
    corpus = cleaner.clean(iter_source_documents(config.data.raw_paths))
    tokenizer = train_tokenizer(config.tokenizer, corpus, args.tokenizer_dir)
    logger.info(
        "trained '%s' tokenizer: vocab_size=%d version=%s -> %s",
        tokenizer.name,
        tokenizer.vocab_size,
        tokenizer.version,
        args.tokenizer_dir,
    )
    print(f"vocab_size={tokenizer.vocab_size} version={tokenizer.version}")
    return 0


def cmd_tokenizer_test(args: argparse.Namespace) -> int:
    from fontaine.tokenizer.registry import load_tokenizer

    tokenizer = load_tokenizer(args.tokenizer_dir)
    text = args.text or "The quick brown fox jumps over the lazy dog."
    ids = tokenizer.encode(text, add_special_tokens=True)
    decoded = tokenizer.decode(ids)
    print(f"tokenizer: {tokenizer.name} (vocab_size={tokenizer.vocab_size}, version={tokenizer.version})")
    print(f"text:      {text!r}")
    print(f"ids:       {ids}")
    print(f"decoded:   {decoded!r}")
    if decoded != text:
        logger.warning("round-trip mismatch (expected for lossy vocabularies)")
    return 0


# -- model ---------------------------------------------------------------------


def cmd_model_inspect(args: argparse.Namespace) -> int:
    from fontaine.optimization import (
        estimate_inference_memory,
        estimate_parameter_count,
        estimate_training_memory,
        format_bytes,
    )

    config = _load(args)
    if isinstance(config.model.vocab_size, str):
        if args.tokenizer_dir:
            from fontaine.tokenizer.registry import load_tokenizer, resolve_vocab_size

            config.model.vocab_size = resolve_vocab_size(
                config.model, load_tokenizer(args.tokenizer_dir)
            )
        else:
            # Nominal vocab for estimation only; pass --tokenizer-dir for exact numbers.
            config.model.vocab_size = 8192
            print(
                "note: model.vocab_size=auto — using nominal 8192 for estimates "
                "(pass --tokenizer-dir for exact numbers)"
            )
    counts = estimate_parameter_count(config.model)
    print(f"architecture:          {config.model.architecture}")
    print(f"hidden_size:           {config.model.hidden_size}")
    print(f"num_layers:            {config.model.num_layers}")
    print(f"attention heads (q/k): {config.model.num_attention_heads}/{config.model.num_kv_heads}")
    print(f"max_sequence_length:   {config.model.max_sequence_length}")
    print(f"parameters (total):    {counts['total']:,}")
    print(f"parameters (active):   {counts['active']:,}")
    print(f"parameters (non-emb):  {counts['non_embedding']:,}")
    if config.model.cpu_tier:
        print(f"cpu tier:              {config.model.cpu_tier}")
    if config.model.num_experts > 1:
        print(
            f"experts:               {config.model.num_experts} "
            f"(top-{config.model.num_experts_per_token} per token)"
        )
    print()
    print("estimated training memory (fp32 + AdamW), per micro-batch:")
    for batch_size in (1, config.training.batch_size):
        estimate = estimate_training_memory(
            config.model,
            batch_size=batch_size,
            sequence_length=config.data.sequence_length,
            gradient_checkpointing=config.training.gradient_checkpointing,
        )
        print(f"  batch_size={batch_size}: " + ", ".join(f"{k}={v}" for k, v in estimate.as_dict().items()))
    print()
    print("rule of thumb: parameters + gradients + AdamW states are about 16 bytes/parameter "
          f"({format_bytes(16 * counts['total'])} here, using total parameters) "
          "BEFORE activations.")
    print()
    print("estimated inference memory (full context window):")
    for precision in ("fp32", "int8"):
        serving = estimate_inference_memory(config.model, precision)
        print(
            f"  {precision}: weights={format_bytes(serving['weights'])}, "
            f"kv_cache={format_bytes(serving['kv_cache'])}, total={format_bytes(serving['total'])}"
        )
    return 0


def cmd_model_recommend(args: argparse.Namespace) -> int:
    from fontaine.config.loader import load_fontaine_config
    from fontaine.optimization import (
        estimate_inference_memory,
        estimate_parameter_count,
        format_bytes,
    )
    from fontaine.optimization.hardware import (
        TIER_CONFIGS,
        YAMI_TIERS,
        detect_hardware,
        recommend_tier,
    )

    info = detect_hardware()
    tier = recommend_tier(info)
    ram = f"{info.total_ram_gib:.1f} GiB" if info.total_ram_gib is not None else "unknown"
    print(f"cpu cores (logical):   {info.logical_cores or 'unknown'}")
    print(f"ram:                   {ram}")
    print(f"cpu capability:        {info.cpu_capability}")
    print(f"native bf16:           {'yes' if info.bf16 else 'no'}")
    print(f"cuda:                  {'yes' if info.cuda else 'no'}")
    print()
    root = Path(__file__).resolve().parents[3]
    for name in YAMI_TIERS:
        path = root / TIER_CONFIGS[name]
        if not path.is_file():
            path = Path(TIER_CONFIGS[name])
        if not path.is_file():
            print(f"  {name:<6} (config {TIER_CONFIGS[name]} not found)")
            continue
        model = load_fontaine_config([str(path)]).model
        params = estimate_parameter_count(model)["total"]
        serving = estimate_inference_memory(model, "int8")
        marker = "  <- recommended" if name == tier else ""
        print(
            f"  {name:<6} {params / 1e6:7.0f}M params, int8 serving about "
            f"{format_bytes(serving['total'])}{marker}"
        )
    print()
    print(f"recommended tier: {tier} ({TIER_CONFIGS[tier]})")
    print(f"train with:       --model-config {TIER_CONFIGS[tier]}")
    print("serve with:       --precision auto (int8 weights on CPU)")
    return 0


# -- train ---------------------------------------------------------------------


def resolve_resume_target(resume_path: Path) -> tuple[Path, str | Path]:
    """Map a ``--resume`` path to ``(experiment_dir, checkpoint_source)``.

    ``experiments/<run>/checkpoints`` holds ``latest.json``. The experiment
    directory is that folder's parent. A ``step_*`` directory holds
    ``meta.json``; the experiment directory is the parent of ``checkpoints``.
    """
    resume_path = Path(resume_path)
    if (resume_path / "latest.json").is_file():
        return resume_path.parent, "latest"
    if (resume_path / "best.json").is_file():
        return resume_path.parent, "best"
    if (resume_path / "meta.json").is_file():
        return resume_path.parent.parent, resume_path
    raise ConfigError(f"--resume {resume_path} is not a checkpoints directory")


def dataloader_workers(device) -> int:
    """Two workers once the run is on CUDA. CPU and MPS stay on the main process."""
    return 2 if device.type == "cuda" else 0


def cmd_train(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader

    from fontaine.data import TokenShardDataset
    from fontaine.data.manifest import DatasetManifest
    from fontaine.distributed.runtime import build_trainable_model
    from fontaine.tokenizer.registry import load_tokenizer, resolve_vocab_size
    from fontaine.training import ExperimentRun, Trainer

    config = _load(args)
    if config.training.stage in ("preference_training", "reinforcement_learning"):
        from fontaine.training.stages import run_post_training

        return run_post_training(config, args.tokenizer_dir)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    # "auto" adopts the tokenizer's vocab size; explicit values must match it.
    config.model.vocab_size = resolve_vocab_size(config.model, tokenizer)

    # Prepare the dataset if no manifest exists yet (end-to-end convenience).
    if config.data.manifest_path is None:
        from fontaine.data import prepare_dataset

        result = prepare_dataset(config.data, tokenizer, name=args.dataset_name)
        config.data.manifest_path = str(result.output_dir / "manifest.json")
    manifest_dir = Path(config.data.manifest_path).parent
    DatasetManifest.load(manifest_dir).validate(manifest_dir, verify_checksums=False)

    train_dataset = TokenShardDataset(manifest_dir, "train", config.data.sequence_length)
    eval_dataset = None
    if "val" in DatasetManifest.load(manifest_dir).splits:
        eval_dataset = TokenShardDataset(manifest_dir, "val", config.data.sequence_length)
    if len(train_dataset) == 0:
        raise ConfigError(
            "training dataset is empty — add more data or reduce data.sequence_length"
        )

    from fontaine.optimization import resolve_device

    num_workers = dataloader_workers(resolve_device(config.training.device))
    train_loader = DataLoader(
        train_dataset, batch_size=config.training.batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    eval_loader = (
        DataLoader(eval_dataset, batch_size=config.training.batch_size, drop_last=False)
        if eval_dataset is not None
        else None
    )
    if eval_loader is None:
        logger.warning("no validation split; evaluation metrics will be unavailable")

    model = build_trainable_model(config)

    resume_dir = None
    resume_source = None
    if args.resume:
        resume_dir, resume_source = resolve_resume_target(Path(args.resume))
        logger.info("resuming from %s (%s)", resume_source, resume_dir)

    run = ExperimentRun.create(
        config.training.experiments_dir,
        config.training.run_name,
        config,
        resume_dir=resume_dir,
    )
    logger.info(
        "training %s: %.2fM active / %.2fM total | %d train windows | device config=%s",
        config.training.run_name,
        model.num_active_parameters() / 1e6,
        model.num_parameters() / 1e6,
        len(train_dataset),
        config.training.device,
    )
    trainer = Trainer(config, model, tokenizer, train_loader, eval_loader, run)
    final_metrics = trainer.fit(resume=resume_source)
    print(json.dumps({"final_metrics": final_metrics, "experiment_dir": str(run.directory)}, indent=2))
    return 0


# -- evaluate --------------------------------------------------------------------


def cmd_evaluate(args: argparse.Namespace) -> int:
    from fontaine.data import TokenShardDataset
    from fontaine.data.manifest import DatasetManifest
    from fontaine.evaluation import EvalContext, build_evaluator
    from fontaine.inference import load_generator

    config = _load(args)
    # Metrics stay comparable with training when weights are not quantized.
    config.inference.precision = "fp32"
    generator = load_generator(
        args.checkpoint, args.tokenizer_dir, config.inference, device="cpu",
        distributed=config.distributed,
    )
    context = EvalContext(
        device="cpu",
        tokenizer=generator.tokenizer,
        generator=generator,
        inference_config=config.inference,
    )
    manifest_dir = None
    if config.data.manifest_path:
        manifest_dir = Path(config.data.manifest_path).parent
    else:
        candidate = Path(config.data.output_dir) / "manifest.json"
        manifest_dir = candidate.parent if candidate.is_file() else None
    if manifest_dir is not None and DatasetManifest.load(manifest_dir).splits.get(config.evaluation.split):
        dataset = TokenShardDataset(
            manifest_dir, config.evaluation.split, generator.model.config.max_sequence_length
        )
        from torch.utils.data import DataLoader

        context.eval_loader = DataLoader(dataset, batch_size=config.training.batch_size, drop_last=False)

    results: dict[str, dict[str, float]] = {}
    for entry in config.evaluation.evaluators:
        evaluator = build_evaluator(entry, context)
        results[evaluator.name] = evaluator.run(generator.model, context)
    print(json.dumps(results, indent=2))
    return 0


# -- generate --------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> int:
    from fontaine.inference import load_generator

    config = _load(args)
    _apply_runtime_flags(config, args)
    generator = load_generator(
        args.checkpoint, args.tokenizer_dir, config.inference, device=args.device,
        distributed=config.distributed,
    )
    if args.interactive:
        print("interactive generation — empty line exits")
        while True:
            try:
                prompt = input("prompt> ")
            except EOFError:
                break
            if not prompt:
                break
            _print_stream(generator, prompt)
        return 0

    prompt = args.prompt or ""
    if not prompt:
        raise ConfigError("provide --prompt or use --interactive")
    _print_stream(generator, prompt, max_new_tokens=args.max_new_tokens)
    return 0


def _apply_runtime_flags(config, args: argparse.Namespace) -> None:
    """Let ``--precision`` and ``--threads`` override the inference config."""
    if args.precision is not None:
        config.inference.precision = args.precision
    if args.threads is not None:
        config.inference.num_threads = args.threads
    config.inference.validate()


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16", "int8"],
        default=None,
        help="weight format (default: inference.precision; auto = int8 on CPU for 20M+ params)",
    )
    parser.add_argument(
        "--threads", type=int, default=None, help="CPU threads (0 = automatic)"
    )


def _print_stream(generator, prompt: str, max_new_tokens: int | None = None) -> None:
    chunks = generator.generate(prompt, stream=True, max_new_tokens=max_new_tokens)
    for chunk in chunks:  # type: ignore[union-attr]
        print(chunk, end="", flush=True)
    print()


# -- checkpoint --------------------------------------------------------------------


def cmd_checkpoint_inspect(args: argparse.Namespace) -> int:
    from fontaine.checkpointing import CheckpointManager

    path = Path(args.checkpoint)
    if (path / "latest.json").is_file():
        manager = CheckpointManager(path)
        meta = manager.inspect("latest")
    else:
        manager = CheckpointManager(path.parent)
        meta = manager.inspect(path)
    print(json.dumps(meta, indent=2))
    return 0


def cmd_checkpoint_convert(args: argparse.Namespace) -> int:
    from fontaine.checkpointing.convert import convert_checkpoint

    out = convert_checkpoint(args.checkpoint, args.format, args.output)
    print(f"converted checkpoint written to {out}")
    return 0


# -- serve --------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    from fontaine.inference import load_generator, serve

    config = _load(args)
    _apply_runtime_flags(config, args)
    generator = load_generator(
        args.checkpoint, args.tokenizer_dir, config.inference, device=args.device,
        distributed=config.distributed,
    )
    serve(
        generator,
        host=config.inference.server_host,
        port=config.inference.server_port,
        model_name=config.inference.model_name,
    )
    return 0


def _numeric_model(config):
    if isinstance(config.model.vocab_size, str):
        config.model.vocab_size = 8192
        print("note: model.vocab_size=auto — using nominal 8192 for estimates")
    return config


def cmd_model_validate(args: argparse.Namespace) -> int:
    from fontaine.models.registry import registry_record

    config = _numeric_model(_load(args))
    record = registry_record(config.model, precision=config.training.precision)
    print("config OK")
    print(json.dumps(record, indent=2))
    return 0


def cmd_model_estimate(args: argparse.Namespace) -> int:
    from fontaine.optimization.memory import (
        estimate_inference_memory,
        estimate_kv_cache_bytes,
        estimate_parameter_count,
        estimate_training_memory,
        format_bytes,
    )

    config = _numeric_model(_load(args))
    counts = estimate_parameter_count(config.model)
    labels = [
        ("Embedding parameters", "embedding"),
        ("Attention parameters", "attention"),
        ("Q parameters", "q"),
        ("K parameters", "k"),
        ("V parameters", "v"),
        ("O projection parameters", "o"),
        ("Dense FFN parameters", "dense_ffn"),
        ("MoE expert parameters", "moe_experts"),
        ("Router parameters", "router"),
        ("Shared expert parameters", "shared_experts"),
        ("Normalization parameters", "normalization"),
        ("LM head parameters", "lm_head"),
        ("Total parameters", "total"),
        ("Active parameters per token", "active"),
    ]
    for label, key in labels:
        print(f"{label}: {counts[key]:,}")
    training = estimate_training_memory(
        config.model,
        batch_size=config.training.batch_size,
        sequence_length=min(config.data.sequence_length, config.model.max_sequence_length),
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    inference = estimate_inference_memory(config.model, "fp32")
    print(f"Estimated optimizer memory: {format_bytes(training.optimizer_bytes)}")
    print(f"Estimated parameter memory: {format_bytes(training.parameters_bytes)}")
    print(f"Estimated activation memory: {format_bytes(training.activations_bytes)}")
    print(
        "Estimated KV-cache memory: "
        f"{format_bytes(estimate_kv_cache_bytes(config.model))} "
        f"(inference, full context; not added to the training total)"
    )
    print(f"Approximate total training memory: {format_bytes(training.total_bytes)}")
    if counts["active"] != counts["total"]:
        print("active parameters differ from total parameters")
    print(f"Inference weight memory (fp32): {format_bytes(inference['weights'])}")
    return 0


def cmd_model_feasibility(args: argparse.Namespace) -> int:
    from fontaine.optimization.feasibility import assess
    from fontaine.optimization.hardware import detect_hardware

    config = _numeric_model(_load(args))
    report = assess(config, detect_hardware(), world_size=args.world_size)
    print("\n".join(report.lines()))
    return 0


def cmd_hardware_inspect(args: argparse.Namespace) -> int:
    from fontaine.optimization.hardware import detect_hardware
    from fontaine.optimization.memory import format_bytes

    info = detect_hardware()
    ram = format_bytes(info.total_ram_bytes) if info.total_ram_bytes else "unknown"
    available = format_bytes(info.available_ram_bytes) if info.available_ram_bytes else "unknown"
    print(f"CPU: {info.cpu_name or 'unknown'}")
    print(f"CPU capability: {info.cpu_capability}")
    print(f"logical cores: {info.logical_cores or 'unknown'}")
    print(f"RAM: {ram}")
    print(f"available system memory: {available}")
    print(f"GPU count: {info.gpu_count}")
    if info.gpus:
        for gpu in info.gpus:
            print(f"GPU {gpu.index}: {gpu.name}  VRAM {format_bytes(gpu.memory_bytes)}")
    else:
        print("GPU model: none")
        print("VRAM: none")
    print(f"CUDA availability: {'yes' if info.cuda else 'no'}")
    print(f"accelerator backend: {info.backend}")
    print(f"interconnect: {info.interconnect}")
    print(f"supported precision: {', '.join(info.precisions)}")
    print(f"available attention kernels: {', '.join(info.attention_kernels)}")
    return 0


def cmd_distributed_plan(args: argparse.Namespace) -> int:
    from fontaine.distributed.topology import plan_topology
    from fontaine.optimization.hardware import detect_hardware

    config = _numeric_model(_load(args))
    info = detect_hardware()
    world = args.world_size or info.gpu_count or 1
    topology = plan_topology(config.model, world)
    print(f"world size: {topology.world_size}")
    print(f"Data Parallel: {topology.data_parallel}")
    print(f"Tensor Parallel: {topology.tensor_parallel}")
    print(f"Pipeline Parallel: {topology.pipeline_parallel}")
    print(f"Sequence Parallel: {topology.sequence_parallel}")
    print(f"Expert Parallel: {topology.expert_parallel}")
    print("This plan was not launched. Pass the sizes in the distributed section and start torchrun yourself.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from fontaine.optimization.benchmark import run_benchmarks

    config = _numeric_model(_load(args))
    result = run_benchmarks(config.model, allow_large=args.allow_large)
    print(json.dumps(result, indent=2))
    return 0


# -- parser assembly ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fontaine",
        description="Fontaine AI: a modular foundation-model platform.",
    )
    parser.add_argument("--version", action="version", version=f"fontaine {__version__}")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="dataset validation and preparation")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    p = data_sub.add_parser("validate", help="verify raw inputs and/or prepared manifest")
    _add_config_arguments(p)
    p.add_argument("--skip-checksums", action="store_true")
    p.set_defaults(func=cmd_data_validate)

    p = data_sub.add_parser("prepare", help="raw data -> token shards + manifest")
    _add_config_arguments(p)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--name", default="fontaine-dataset")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--license", default=None, help="dataset license identifier (SPDX preferred)")
    p.add_argument("--language", default=None)
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_data_prepare)

    tokenizer = sub.add_parser("tokenizer", help="tokenizer training and testing")
    tok_sub = tokenizer.add_subparsers(dest="tokenizer_command", required=True)
    p = tok_sub.add_parser("train", help="train a tokenizer on the configured corpus")
    _add_config_arguments(p)
    p.add_argument("--tokenizer-dir", required=True)
    p.set_defaults(func=cmd_tokenizer_train)

    p = tok_sub.add_parser("test", help="encode/decode round-trip smoke test")
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--text", default=None)
    p.set_defaults(func=cmd_tokenizer_test)

    model = sub.add_parser("model", help="model inspection")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    p = model_sub.add_parser("inspect", help="parameter counts + memory estimates")
    _add_config_arguments(p)
    p.add_argument("--tokenizer-dir", default=None, help="resolve vocab_size=auto exactly")
    p.set_defaults(func=cmd_model_inspect)
    p = model_sub.add_parser("recommend", help="pick a Yami size tier for this machine")
    p.set_defaults(func=cmd_model_recommend)
    p = model_sub.add_parser("validate", help="validate a config without building the model")
    _add_config_arguments(p)
    p.set_defaults(func=cmd_model_validate)
    p = model_sub.add_parser("estimate-params", help="parameter and memory estimates")
    _add_config_arguments(p)
    p.set_defaults(func=cmd_model_estimate)
    p = model_sub.add_parser("feasibility", help="compare a config to this machine")
    _add_config_arguments(p)
    p.add_argument("--world-size", type=int, default=None)
    p.set_defaults(func=cmd_model_feasibility)

    hardware = sub.add_parser("hardware", help="machine inspection")
    hardware_sub = hardware.add_subparsers(dest="hardware_command", required=True)
    p = hardware_sub.add_parser("inspect", help="CPU, RAM, GPU, precision, attention kernels")
    p.set_defaults(func=cmd_hardware_inspect)

    distributed = sub.add_parser("distributed", help="parallelism planning")
    distributed_sub = distributed.add_subparsers(dest="distributed_command", required=True)
    p = distributed_sub.add_parser("plan", help="suggest a topology; does not launch training")
    _add_config_arguments(p)
    p.add_argument("--world-size", type=int, default=None)
    p.set_defaults(func=cmd_distributed_plan)

    benchmark = sub.add_parser("benchmark", help="time a tiny forward and backward")
    _add_config_arguments(benchmark)
    benchmark.add_argument("--allow-large", action="store_true")
    benchmark.set_defaults(func=cmd_benchmark)

    train = sub.add_parser("train", help="train a model")
    _add_config_arguments(train)
    train.add_argument("--tokenizer-dir", required=True)
    train.add_argument("--dataset-name", default="fontaine-dataset")
    train.add_argument("--resume", default=None, help="checkpoints dir (latest) or step dir")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="run evaluators against a checkpoint")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--tokenizer-dir", required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(func=cmd_evaluate)

    generate = sub.add_parser("generate", help="generate text from a checkpoint")
    _add_config_arguments(generate)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--tokenizer-dir", required=True)
    generate.add_argument("--prompt", default="")
    generate.add_argument("--max-new-tokens", type=int, default=None)
    generate.add_argument("--interactive", action="store_true")
    generate.add_argument("--device", default="auto")
    _add_runtime_arguments(generate)
    generate.set_defaults(func=cmd_generate)

    checkpoint = sub.add_parser("checkpoint", help="checkpoint utilities")
    ckpt_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    p = ckpt_sub.add_parser("inspect", help="show checkpoint metadata")
    p.add_argument("--checkpoint", required=True)
    p.set_defaults(func=cmd_checkpoint_inspect)

    p = ckpt_sub.add_parser("convert", help="convert checkpoints between formats")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--format", default="safetensors", choices=["safetensors"])
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_checkpoint_convert)

    serve = sub.add_parser("serve", help="start the local inference HTTP API")
    _add_config_arguments(serve)
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--tokenizer-dir", required=True)
    serve.add_argument("--device", default="auto")
    _add_runtime_arguments(serve)
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=10 if args.verbose else 20)
    try:
        result = args.func(args)
        return int(result)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
