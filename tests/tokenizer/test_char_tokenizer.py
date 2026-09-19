"""Tokenizer subsystem tests."""

import pytest

from fontaine.tokenizer import CharTokenizer, SpecialTokens


def test_special_tokens_occupy_first_four_ids():
    tok = CharTokenizer.train(["abc"])
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2 and tok.unk_id == 3
    assert tok.vocab_size == 4 + len(set("abc"))


def test_roundtrip_and_unknown_chars():
    tok = CharTokenizer.train(["abc"])
    ids = tok.encode("abc", add_special_tokens=True)
    assert ids == [tok.bos_id, 4, 5, 6, tok.eos_id]
    assert tok.decode(ids, skip_special_tokens=True) == "abc"
    unknown = tok.encode("abcxyz")
    assert unknown[3] == tok.unk_id


def test_save_load_preserves_vocab_and_version(tmp_path):
    tok = CharTokenizer.train(["hello world", "abc"], lowercase=True)
    version = tok.version
    tok.save(tmp_path / "tok")
    loaded = CharTokenizer.load(tmp_path / "tok")
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.version == version
    assert loaded.encode("Hello") == tok.encode("Hello")


def test_version_changes_with_vocab():
    tok_a = CharTokenizer.train(["abc"])
    tok_b = CharTokenizer.train(["abcd"])
    assert tok_a.version != tok_b.version


def test_lowercase_training():
    tok = CharTokenizer.train(["ABC abc"], lowercase=True)
    # 'A' is not in the corpus; the lowercase form 'a' is.
    assert tok.encode("A") == [tok.unk_id]
    assert tok.encode("a") != [tok.unk_id]


def test_encode_batch():
    tok = CharTokenizer.train(["abc def"])
    batch = tok.encode_batch(["abc", "def"])
    assert batch == [tok.encode("abc"), tok.encode("def")]


def test_corpus_chars_must_not_collide_with_specials():
    # Custom single-character special tokens cannot double as corpus chars.
    specials = SpecialTokens(pad="|", bos="<", eos=">", unk="_")
    with pytest.raises(ValueError, match="collide"):
        CharTokenizer(sorted("a|b"), specials)


def test_custom_single_char_specials_work():
    specials = SpecialTokens(pad="|", bos="<", eos=">", unk="_")
    tok = CharTokenizer(sorted("abc"), specials)
    assert tok.pad_id == 0 and tok.bos_id == 1
    assert tok.encode("a|b<c") == [4, 0, 5, 1, 6]


def test_bpe_train_encode_decode_roundtrip():
    tok = pytest.importorskip("fontaine.tokenizer.hf_bpe").ByteLevelBPETokenizer.train(
        ["the quick brown fox " * 10, "jumps over the lazy dog " * 10] * 5,
        vocab_size=300,
        min_frequency=1,
    )
    text = "the quick brown fox"
    ids = tok.encode(text)
    assert ids and all(0 <= i < tok.vocab_size for i in ids)
    assert tok.decode(ids) == text
    assert tok.decode(tok.encode(text, add_special_tokens=True), skip_special_tokens=True) == text


def test_bpe_save_load(tmp_path):
    hf_bpe = pytest.importorskip("fontaine.tokenizer.hf_bpe").ByteLevelBPETokenizer
    tok = hf_bpe.train(["fontaine flows " * 30], vocab_size=280, min_frequency=1)
    tok.save(tmp_path / "bpe")
    loaded = hf_bpe.load(tmp_path / "bpe")
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("fontaine") == tok.encode("fontaine")
    assert loaded.version == tok.version
