def is_valid(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in s:
        if char in pairs.values():
            stack.append(char)
        elif not stack or stack.pop() != pairs[char]:
            return False
    return not stack
