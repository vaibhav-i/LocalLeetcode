from __future__ import annotations


def contains_duplicate(nums: list[int]) -> bool:
    return len(set(nums)) != len(nums)
