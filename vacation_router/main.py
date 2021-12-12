# %%
from monty.serialization import loadfn
import numpy as np
import pandas as pd
from vacation_router.gen_algo import get_best_perturbed_masks
# %%
best_path_for_bm = loadfn("best_path_for_bm.json")
valid_bitmasks = set(best_path_for_bm.keys())
df = pd.read_csv("./parsed_df.csv")

# %%

scores = dict(enumerate(df.Total))

def get_score(bm):
    # calculate a score for a bitmask
    score = 0
    for i in range(bm.bit_length()):
        if bm & (1 << i):
            score += scores[i]
    return score

# %%
# randomly select 4 bitmasks from valid_bitmasks
bitmasks = np.random.choice(list(valid_bitmasks), 3, replace=False)
# %%
get_best_perturbed_masks(bitmasks, valid_bitmasks, get_score)
