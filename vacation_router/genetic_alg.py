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

_logger = logging.getLogger(__name__)

def pertub_bitmask(bitmask: int, lam:float = 2.0, belong_to: set = None, exclude_bits: int = None):
    """
    Randomly flip bits in a bitmask use a poisson distribution for the number of bits to flip
    Args:
        bitmask: a bitmask 
        lam: the lambda parameter for the poisson distribution
        belong_to: a set of bitmasks that the bitmask must belong to
        exclude_bits: a set of bitmasks that the bitmask must not belong to
    Returns:
        a new bitmask
    """
    def pert_once(bm):
        num_bits = np.random.poisson(lam=lam)
        # randomly select bits to flip
        bits_to_flip = np.random.choice(range(len(bin(bm))-2), num_bits, replace=False)

        for bit in bits_to_flip:
            bm ^= 1 << bit
        return bm 

    if belong_to is None and exclude_bits is None:
        return pert_once(bitmask)
    else:
        for i in range(100000):
            bm = pert_once(bitmask)
            # _logger.debug(f"bm: {bm:#032b}, exclude: {exclude_bits:#032b}")
            if (belong_to is None or bm in belong_to) and (exclude_bits is None or bm & exclude_bits == 0):
                return bm
        else:
            raise Exception("Failed to pertub bitmask")

def get_valid_puturbed_masks(bitmasks, valid_bitmasks, lam):
    """
    Starting for a set of bitmasks, perturb them and find a valid combination that does not clash
    """
    updated_masks = []
    seent_bits = 0
    for bm in bitmasks:
        new_mask = pertub_bitmask(bm, belong_to=valid_bitmasks, exclude_bits=seent_bits, lam=lam)
        _logger.debug(f"new_mask: {new_mask:#032b}")
        seent_bits |= new_mask
        updated_masks.append(new_mask)
    return updated_masks

def get_best_perturbed_masks(bitmasks: int, 
                    valid_bitmasks: set, 
                    lam: float,
                    score_func: Callable, 
                    high_score: bool=True):
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
    randomized_masks = get_valid_puturbed_masks(bitmasks[:-1], set(valid_bitmasks), lam=lam)
    best_final_mask = None
    all_masked_bits = reduce(lambda x, y: x | y, randomized_masks)
    best_score = -float("inf")
    for bm in valid_bitmasks:
        score_ = score_func(bm) 
        if not  high_score:
            score_ *= -1
        if all_masked_bits & bm == 0 and score_ > best_score:
            best_score = score_
            best_final_mask = bm
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
    
    def _get_mask(self):
        bms = np.random.choice(self.valid_bitmasks, self.n_masks, replace=False)
        return get_best_perturbed_masks(bms, self.valid_bitmasks, self.score_func, self.high_score)
    
    def __post_init__(self):
        self.factor = self.n_masks // self.n_keep
        self.n_masks  = self.factor * self.n_keep
        self.pop = [self._get_mask() for _ in range(self.pop_size)]
    
    def get_most_fit(self):
        """
        Get the top most fit bitmasks
        """
        return sorted(self.pop, key=lambda x: self.score_func(x[-1]), reverse=self.high_score)[:self.n_keep]

    def mutate(self, bitmasks: List[int]):
        """
        Mutate a list of bitmasks
        Args:
            bitmasks: a list of bitmasks
        """
        return get_best_perturbed_masks(bitmasks, self.valid_bitmasks, self.lam, self.score_func, self.high_score)

    def step(self):
        """
        Step the genetic algorithm
        """
        bms = self.get_most_fit()
        bms *= chain.from_iterable(bms * self.factor)
        print(bms)

    
        


    



# def fitness(bitmask: int):
#     pass

# def select_parents(population: list):
#     pass

# def generate_population(population_size: int):
#     pass

# def genetic_algorithm(population_size: int, num_generations: int):
#     pass


# # %%
# if __name__ == '__main__':
#     least_time_json = loadfn("least_time.json")
#     least_time_json = {eval(k): v for k, v in least_time_json.items()}
#     parent_json = loadfn("parent.json")
#     parent_json = {eval(k): tuple(v) for k, v in parent_json.items()}
#     least_time = defaultdict(lambda: float("inf"))
#     parent = defaultdict(lambda: None)
#     least_time.update(least_time_json)
#     parent.update(parent_json)
#     # %%
#     best_time_for_bitmask = defaultdict(lambda: float('inf'))
#     best_seq_for_bitmask = defaultdict(lambda: None)
#     for (node, bit_mask), best_time in tqdm(least_time.items()):
#         if best_time == float('inf'):
#             continue
#         if best_time < best_time_for_bitmask[bit_mask]:
#             best_time_for_bitmask[bit_mask] = best_time
#             best_seq_for_bitmask[bit_mask] = get_sequence(parent, (node, bit_mask))
#     valid_bitmasks = set(best_seq_for_bitmask.keys())
# # %%
# best_time_for_bitmask[4198928]
# # %%
# best_seq_for_bitmask[4198928]
# # %%
# # randomly select 4 bitmasks that do not share a common bit
# bitmasks = [np.random.randint(0, 2**32) for _ in range(4)]


# # %%
# for _ in range(10):
#     print(np.random.poisson(lam=5.5))


# # %%
# # printbm(4198928)
# # printbm(pertub_bitmask(4198928, belong_to=valid_bitmasks) ^ 4198928)
# # printbm(pertub_bitmask(4198928, belong_to=valid_bitmasks) ^ 4198928)
# # printbm(pertub_bitmask(4198928, belong_to=valid_bitmasks) ^ 4198928)

# # randomly select 4 bitmasks from valid_bitmasks
# bitmasks = np.random.choice(list(valid_bitmasks), 3, replace=False)


# for bm in get_best_perturbed_masks(bitmasks, best_time_for_bitmask):
#     print(f"{bm:014d} -> {bm:#032b}")

    


# # for bm in bitmasks:
# #     pertub_bitmask(bitmasks)
# # %%
# logging.basicConfig(level=logging.INFO)
# print(get_valid_triple(bitmasks))
# # %%

# def get_sequence(parent, idx):
#     seq = []
#     while idx != None:
#         seq.append(idx[0])
#         idx = parent[idx]
#     return seq[::-1]
# %%
