"""Pack tokenized documents into fixed windows and measure padding waste."""

from fontaine.tokenizer.base import Tokenizer


def pack_token_documents(
    documents: list[list[int]],
    sequence_length: int,
    eos_id: int,
) -> tuple[list[list[int]], list[list[int]], int]:
    """Fill windows of ``sequence_length`` tokens. No padding is written.

    Each window has a parallel list of document ids so attention can stop at
    document boundaries. The return value is windows, document ids, and the
    number of leftover tokens that did not fill a window.
    """
    if sequence_length < 2:
        raise ValueError(f"sequence_length must be >= 2, got {sequence_length}")
    windows: list[list[int]] = []
    doc_ids: list[list[int]] = []
    current: list[int] = []
    owners: list[int] = []
    for doc_index, tokens in enumerate(documents):
        piece = list(tokens) + [eos_id]
        for token in piece:
            current.append(token)
            owners.append(doc_index)
            if len(current) == sequence_length:
                windows.append(current)
                doc_ids.append(owners)
                current = []
                owners = []
    return windows, doc_ids, len(current)


def packing_efficiency(document_lengths: list[int], sequence_length: int) -> float:
    """Fraction of tokens that land in a full window when documents are concatenated.

    Padding each document up to ``sequence_length`` is the comparison point:
    efficiency is packed tokens divided by the padded token count.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    total = sum(document_lengths)
    if total == 0:
        return 0.0
    packed = total - (total % sequence_length)
    padded = sum(((length + sequence_length - 1) // sequence_length) * sequence_length for length in document_lengths)
    if padded == 0:
        return 0.0
    return packed / padded


def mask_between_documents(document_ids: list[int]) -> list[list[bool]]:
    """Causal mask that is also false across document ids."""
    size = len(document_ids)
    mask = []
    for query in range(size):
        row = []
        for key in range(size):
            visible = key <= query and document_ids[key] == document_ids[query]
            row.append(visible)
        mask.append(row)
    return mask


def response_mask(token_ids: list[int], marker_ids: list[int]) -> list[int]:
    """Labels equal to the tokens, with positions before the response marker set to -100.

    The marker itself is masked. Tokens after it are trained. If the marker is
    absent, every label is -100 and the caller can see that nothing was supervised.
    """
    start = _find_subsequence(token_ids, marker_ids)
    labels = [-100] * len(token_ids)
    if start is None:
        return labels
    content = start + len(marker_ids)
    for index in range(content, len(token_ids)):
        labels[index] = token_ids[index]
    return labels


def _find_subsequence(haystack: list[int], needle: list[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    last = len(haystack) - len(needle) + 1
    for start in range(last):
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def encode_marker(tokenizer: Tokenizer, marker: str = "### Response:\n") -> list[int]:
    return tokenizer.encode(marker, add_special_tokens=False)
