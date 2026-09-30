"""Admission rules for a resident expert tier.

These match the placement policy used by engines that keep a hot subset of
routed experts in fast memory and leave the rest on a colder tier: a saturated
heat counter stays saturated, promotion needs a margin so two close experts do
not swap forever, and frequency outweighs recency.
"""

_U32 = 0xFFFFFFFF


def tier_should_promote(hot: int, cold: int) -> bool:
    """True when ``hot`` is far enough above ``cold`` to take its slot.

    The margin is ``cold / 4 + 4``. A lead of a few hits is not enough.
    """
    threshold = cold + (cold >> 2) + 4
    return hot > threshold


def tier_decay_value(heat: int) -> int:
    """Halve a heat counter. Zero stays zero."""
    return heat >> 1


def tier_lfru_score(heat: int, last: int, clock: int) -> int:
    """Frequency first, recency only as a tie break.

    One frequency count is worth 256. A recent access adds at most 255, so a
    merely recent expert cannot outrank a genuinely hotter one.
    """
    age = (clock - last) & _U32
    recent = 255 - age if age < 255 else 0
    return (heat << 8) | recent


def tier_decay(heat: list[int]) -> None:
    """Halve every counter in ``heat``."""
    for index, value in enumerate(heat):
        heat[index] = tier_decay_value(value)


def tier_pick_swap(
    heat: list[int], pinned: list[int]
) -> tuple[int, int, int] | None:
    """Pick a resident slot to replace, or ``None`` when nobody earns it.

    Returns ``(slot, expert_id, gain)`` where ``slot`` indexes ``pinned``.
    """
    if not heat or not pinned:
        return None
    cold = 0
    for index in range(1, len(pinned)):
        if heat[pinned[index]] < heat[pinned[cold]]:
            cold = index
    resident = set(pinned)
    hot = -1
    hottest = 0
    for expert_id, value in enumerate(heat):
        if expert_id not in resident and value > hottest:
            hottest = value
            hot = expert_id
    if hot < 0:
        return None
    coldest = heat[pinned[cold]]
    if not tier_should_promote(hottest, coldest):
        return None
    return cold, hot, hottest - coldest


def tier_pick_lfru(
    heat: list[int],
    last: list[int],
    clock: int,
    pinned: list[int],
) -> tuple[int, int, int] | None:
    """Like :func:`tier_pick_swap`, but recency breaks close heat values.

    The same 25% plus 4-count margin is applied in score units.
    """
    if not heat or not last or not pinned or len(heat) != len(last):
        return None
    cold = 0
    cold_score = tier_lfru_score(heat[pinned[0]], last[pinned[0]], clock)
    for index in range(1, len(pinned)):
        score = tier_lfru_score(heat[pinned[index]], last[pinned[index]], clock)
        if score < cold_score:
            cold = index
            cold_score = score
    resident = set(pinned)
    hot = -1
    hot_score = 0
    for expert_id in range(len(heat)):
        if expert_id in resident:
            continue
        score = tier_lfru_score(heat[expert_id], last[expert_id], clock)
        if hot < 0 or score > hot_score:
            hot = expert_id
            hot_score = score
    if hot < 0:
        return None
    if hot_score <= cold_score + (cold_score >> 2) + (4 << 8):
        return None
    return cold, hot, (hot_score - cold_score) >> 8
