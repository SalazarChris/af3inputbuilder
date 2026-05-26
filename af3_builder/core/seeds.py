# af3_builder/seeds.py
import random
from typing import List

class SeedsHelper:
    """Utility to generate and validate modelSeeds for AlphaFold3 jobs."""

    @staticmethod
    def validate_seeds(seeds: List[int]) -> None:
        """Validate seeds: must be a non-empty list of integers."""
        if not isinstance(seeds, list) or len(seeds) == 0:
            raise ValueError("modelSeeds must be a non-empty list of integers")
        for s in seeds:
            if not isinstance(s, int):
                raise ValueError(f"Invalid seed: {s}, must be integer")

    @staticmethod
    def generate_random_seeds(n: int, seed_range=(1, 9999)) -> List[int]:
        """Generate n random seeds within the given range."""
        if n <= 0:
            raise ValueError("n must be positive")
        return [random.randint(seed_range[0], seed_range[1]) for _ in range(n)]

    @staticmethod
    def generate_default_seeds(n: int) -> List[int]:
        """Generate n random seeds in the range 1-9999."""
        return SeedsHelper.generate_random_seeds(n)
