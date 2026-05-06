def max_sub_array(nums):
    best = nums[0]
    current = nums[0]
    for value in nums[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best
