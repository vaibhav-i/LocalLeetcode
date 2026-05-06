def product_except_self(nums):
    answer = [1] * len(nums)
    prefix = 1
    for index, value in enumerate(nums):
        answer[index] = prefix
        prefix *= value

    suffix = 1
    for index in range(len(nums) - 1, -1, -1):
        answer[index] *= suffix
        suffix *= nums[index]
    return answer
