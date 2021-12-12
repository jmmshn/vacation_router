#%%
from collections import defaultdict
from functools import reduce
from itertools import chain
from os import get_exec_path
from re import I
from typing import Callable, List
from monty.serialization import loadfn
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass
import random

_logger = logging.getLogger(__name__)


def print_bitmasks(bitmasks):
    for bm in bitmasks:
        print(f"{bm:#032b}")


def perturb_bitmask(
    bitmask: int,
    left_most: int = None,
    lam: float = 2.0,
    belong_to: set = None,
    exclude_mask: int = None,
    max_attempts: int = 10000,
):
    """
    Randomly flip bits in a bitmask use a poisson distribution for the number of bits to flip
    Args:
        bitmask: a bitmask
        lam: the lambda parameter for the poisson distribution
        belong_to: a set of bitmasks that the bitmask must belong to
        exclude_mask: a set of bitmasks that the bitmask must not belong to
    Returns:
        a new bitmask
    """
    if left_most is None:
        left_most = len(bin(bitmask)) - 2

    def pert_once(bm):
        num_bits = np.random.poisson(lam=lam)
        # randomly select bits to flip
        eligible_bits = range(max(left_most, num_bits + 2))
        bits_to_flip = np.random.choice(eligible_bits, num_bits, replace=False)

        for bit in bits_to_flip:
            bm ^= 1 << bit
        return bm

    if belong_to is None and exclude_mask is None:
        return pert_once(bitmask)
    else:
        for _ in range(max_attempts):
            bm = pert_once(bitmask)
            # _logger.debug(f"bm: {bm:#032b}, exclude: {exclude_bits:#032b}")
            if (belong_to is None or bm in belong_to) and (
                exclude_mask is None or bm & exclude_mask == 0
            ):
                return bm
        else:
            raise Exception("Failed to pertub bitmask")


def get_valid_puturbed_masks(
    bitmasks: List[float], valid_bitmasks: set = None, **kwargs
):
    """
    Starting for a set of bitmasks, perturb them and find a valid combination that does not clash
    """
    updated_masks = []
    seent_bits = 0
    for bm in bitmasks:
        new_mask = perturb_bitmask(
            bm, belong_to=valid_bitmasks, exclude_mask=seent_bits, **kwargs
        )
        # _logger.debug(f"seent_bits: {seent_bits:#010b}")
        # _logger.debug(f"old_mask: {bm:#010b}")
        # _logger.debug(f"new_mask: {new_mask:#010b}")
        seent_bits |= new_mask
        updated_masks.append(new_mask)
    return updated_masks


def get_best_perturbed_masks(
    bitmasks: int,
    valid_bitmasks: set,
    score_func: Callable,
    high_score: bool = True,
    **kwargs,
):
    """
    Perturbe the first n-1 masks then loop through the valid bitmasks to find the best final bitmask
    Args:
        bitmasks: a list of bitmasks
        valid_bitmasks: a set of bitmasks that the bitmask must belong to
        lam: the excpected number of bits to flip in a poisson distribution
        score_func: a function that takes a bitmask and returns a score
        high_score: if True, the higher the score the better,
                    if False, the lower the score the better
    Returns:
        the modified bitmasks
    """
    randomized_masks = get_valid_puturbed_masks(
        bitmasks[:-1], set(valid_bitmasks), **kwargs
    )
    if len(randomized_masks) != len(bitmasks) - 1:
        raise Exception("Failed to perturb masks")

    best_final_mask = None
    all_masked_bits = reduce(lambda x, y: x | y, randomized_masks)
    best_score = -float("inf")
    for bm in valid_bitmasks:
        score_ = score_func(bm)
        if not high_score:
            score_ *= -1
        if all_masked_bits & bm == 0 and score_ > best_score:
            best_score = score_
            best_final_mask = bm
    if best_final_mask is None:
        _logger.warning(f"Failed to find best final mask")
        return get_best_perturbed_masks(
            bitmasks=bitmasks,
            valid_bitmasks=valid_bitmasks,
            score_func=score_func,
            high_score=high_score,
            **kwargs,
        )
    _logger.debug(f"all_masked_bits: {all_masked_bits:#032b}")
    _logger.debug(f"best_final_mask: {best_final_mask:#032b}")
    return randomized_masks + [best_final_mask]

@dataclass
class GenePool:
    score_func: Callable
    pop_size: int
    n_masks: int
    lam: float
    valid_bitmasks: set
    high_score: bool
    n_keep: int
    max_attempts: int
    left_most: int

    def __post_init__(self):
        def _get_mask():
            try:
                bms = random.sample(self.valid_bitmasks, self.n_masks)
                new_masks = get_valid_puturbed_masks(
                    bitmasks=bms,
                    valid_bitmasks=set(self.valid_bitmasks),
                    lam=self.lam,
                    left_most=self.left_most,
                )
                return new_masks
            except Exception:
                return _get_mask()
        self.population = [_get_mask() for _ in range(self.pop_size)]

    def fitness(self, bitmasks):
        """
        Calculate the total score of a set of bitmasks
        """
        return sum(map(self.score_func, bitmasks))

    def population_fitness(self):
        """
        Calculate the average fitness of the population
        """
        return np.mean([self.fitness(bm) for bm in self.population])
    
    def most_fit(self):
        """
        Calculate the average fitness of the population
        """
        return np.max([self.fitness(bm) for bm in self.population])

    def step(self):
        """
        Take the most fit members of the population
        Pertube them to arrive at the new population
        """

        def _get_new_masks(masks):
            try:
                return get_valid_puturbed_masks(
                    bitmasks=masks,
                    valid_bitmasks=set(self.valid_bitmasks),
                    lam=self.lam,
                    left_most=self.left_most,
                )
            except Exception:
                return _get_new_masks(masks)

        # get the most fit n_keep members
        self.population.sort(key=self.fitness, reverse=self.high_score)
        self.population = self.population[: self.n_keep]
        _logger.debug(f"population: {self.population}")

        # perturb the batch of most fit members to get the new population
        new_population = []
        while len(new_population) < self.pop_size:
            new_population.extend([*map(_get_new_masks, self.population)])
        # randomely remove the extra members
        self.population = random.sample(new_population, self.pop_size)

# %%
random.seed(0)
genepool = GenePool(
    score_func=lambda x: x,
    pop_size=8,
    n_masks=3,
    lam=1.0,
    valid_bitmasks=range(0b10000000),
    high_score=True,
    n_keep=2,
    max_attempts=1000,
    left_most=None
)
for bm in genepool.population:
    print_bitmasks(bm)
    print("====")
# %%
for i in range(100):
    print(f"{i}: {genepool.most_fit()}")
    genepool.step()
# %%
genepool.population = [[0b1000010, 0b0010000, 0b0000101]]*8
for i in range(5):
    print(f"{i}: {genepool.most_fit()}")
    genepool.step()


# %%
