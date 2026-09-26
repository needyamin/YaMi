"""Generation engine tests: determinism, streaming, stops, cache path."""

from fontaine.config.schema import InferenceConfig
from fontaine.inference import Generator
from fontaine.inference.engine import fit_context
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


def test_full_window_request_keeps_the_prompt(model_config, tokenizer):
    """A request as long as the window must not discard the question or overflow."""
    prompt = "hello"
    prompt_len = len(tokenizer.encode(prompt))
    window = model_config.max_sequence_length
    kept, gen_budget = fit_context(prompt_len, window, window)
    assert kept == prompt_len
    assert gen_budget == window - prompt_len
    assert kept + gen_budget <= window
    config = InferenceConfig(temperature=0.0, max_new_tokens=window)
    generator = Generator(build_model(model_config), tokenizer, config, device="cpu")
    text = generator.generate(prompt)
    assert isinstance(text, str)


def test_context_budget_reserves_prompt_when_both_overflow():
    # 256-token model, 256 requested new tokens, a long prompt: keep 75% for the prompt.
    kept, gen_budget = fit_context(prompt_len=300, max_new_tokens=256, max_sequence_length=256)
    assert kept == 192
    assert gen_budget == 64
    assert kept + gen_budget <= 256


def test_context_budget_keeps_short_prompt_and_shrinks_the_answer():
    kept, gen_budget = fit_context(prompt_len=20, max_new_tokens=256, max_sequence_length=256)
    assert kept == 20
    assert gen_budget == 236
