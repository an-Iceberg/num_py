import sys
from pathlib import Path

sys.path.insert(0, (Path() / "src/").absolute().as_posix())

from fermat_test import is_prime_naive, is_prime_fermat

[print(f"{n}: {is_prime_fermat(n)}") for n in range(2, 300) if is_prime_naive(n)]
