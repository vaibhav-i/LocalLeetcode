from __future__ import annotations


def two_sum_sorted(numbers: list[int], target: int) -> list[int]:
    left = 0
    right = len(numbers) - 1
    while left < right:
        current = numbers[left] + numbers[right]
        if current == target:
            return [left + 1, right + 1]
        if current < target:
            left += 1
        else:
            right -= 1
    raise ValueError("No valid pair found.")
