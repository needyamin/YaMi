"""Generation engine tests: determinism, streaming, stops, cache path."""

from fontaine.config.schema import InferenceConfig
from fontaine.inference import Generator
from fontaine.models import build_model


def _generator(model_config, tokenizer, **inference_overrides) -> Generator:
    config = InferenceConfig(**inference_overrides)
    return Generator(build_model(model_config), tokenizer, config, device="cpu")


def test_greedy_generation_is_deterministic(model_config, tokenizer):
    generator = _generator(model_config, tokenizer, temperature=0.0, max_new_tokens=20)
    a = generator.generate("the fontaine")
    b = generator.generate("the fontaine")
    assert a == b and len(a) > 0


def test_seeded_sampling_is_deterministic(model_config, tokenizer):
    generator = _generator(model_config, tokenizer, temperature=1.0, seed=11, max_new_tokens=20)
    a = generator.generate("stone and water")
    b = generator.generate("stone and water")
    assert a == b


def test_max_new_tokens_respected(model_config, tokenizer):
    generator = _generator(model_config, tokenizer, temperature=0.0, max_new_tokens=15)
    # char tokenizer: ~15 chars generated
    text = generator.generate("the")
    assert len(text) <= 15


def test_stop_sequence_truncates(model_config, tokenizer):
    # the toy corpus is full of " and "; with greedy decoding it appears fast
    generator = _generator(model_config, tokenizer, temperature=0.0, max_new_tokens=60)
    text = generator.generate("the fontaine", stop_sequences=[" and "])
    assert " and " not in text


def test_stream_deltas_join_to_full_text(model_config, tokenizer):
    generator = _generator(model_config, tokenizer, temperature=0.0, max_new_tokens=20)
    streamed = "".join(generator.generate("the fontaine", stream=True))
    full = generator.generate("the fontaine")
    assert streamed == full


def test_stream_deltas_are_incremental(model_config, tokenizer):
    generator = _generator(model_config, tokenizer, temperature=0.0, max_new_tokens=20)
    deltas = list(generator.generate("the fontaine", stream=True))
    assert all(len(d) >= 1 for d in deltas)
    assert sum(len(d) for d in deltas) == len("".join(deltas))


def test_long_prompt_truncated_not_crash(model_config, tokenizer):
    generator = _generator(model_config, tokenizer, temperature=0.0, max_new_tokens=8)
    text = generator.generate("x" * 500)
    assert isinstance(text, str)


def test_empty_generation_at_context_limit(model_config, tokenizer):
    # max_new_tokens fills the whole context: prompt is cut to nothing left
    config = InferenceConfig(temperature=0.0, max_new_tokens=model_config.max_sequence_length)
    generator = Generator(build_model(model_config), tokenizer, config, device="cpu")
    text = generator.generate("hello")
    assert isinstance(text, str)
