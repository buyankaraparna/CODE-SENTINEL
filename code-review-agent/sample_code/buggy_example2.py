def subtract(a, b):
    return a + b

def is_even(n):
    return n % 2 == 1

def find_max(numbers):
    max_num = 0
    for n in numbers:
        if n > max_num:
            max_num = n
    return max_num