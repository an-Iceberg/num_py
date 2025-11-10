# Taken from here: https://stackoverflow.com/questions/8898807/pythonic-way-to-iterate-over-bits-of-integer#answer-8898977
# Helper function
def bits(n: int) -> list[int]:
    """Decomposes a number into a list of its binary factors.
    # Example
    43 = 0b101011 = 0b1 + 0b10 + 0b1000 + 0b100000\\
    0b1 = 1\\
    0b10 = 2\\
    0b1000 = 8\\
    0b100000 = 32\\
    => `[1, 2, 8, 32]`
    """
    bits = []
    while n:
        b = n & (~n + 1)
        bits.append(b)
        n ^= b
    return bits


def log2(factor: int) -> int:
    """From the equation `factor = 2^x` this function calculates x and returns it.

    # ⚠️ Assumption:
    `factor` is a power of 2
    """
    counter = 0
    while factor != 0b1:
        factor >>= 1
        counter += 1
    return counter


def pow_mod(base: int, exponent: int, modulus: int):
    """Calculates a^b mod m"""
    result = 1
    block_result = base
    exponents = [log2(b) for b in bits(exponent)]
    # We don't need to iterate thru k_1 - k_i in order to reach k_i+1,
    # so we skip those previous calculations with this
    previous_k = 0
    for k in exponents:
        for _ in range(k - previous_k):
            block_result **= 2
            block_result %= modulus
        result *= block_result
        result %= modulus
        previous_k = k
    return result
