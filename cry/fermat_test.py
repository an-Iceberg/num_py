from gcd import gcd
from pow_mod import pow_mod
from random import randint


def fermat_test(number: int, a: int) -> bool:
    """
    If this returns `True` then there's a 50% chance that `number` is prime.
    """
    return gcd(number, a) == 1 and pow_mod(a, number - 1, number) == 1


def is_prime_naive(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    return not any([n % i == 0 for i in range(2, n)])


def is_prime_fermat(number: int, iter_count: int = 15):
    """
    Determines if a number is prime using the fermat test.
    This is a monte carlo algorithm.
    """
    if number <= 1:
        return False
    if number == 2 or number == 3:
        return True

    for _ in range(iter_count):
        a = randint(2, number - 2)
        if not fermat_test(number, a):
            return False

    return True
