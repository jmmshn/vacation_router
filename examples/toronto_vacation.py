# %%
from collections import defaultdict
from pathlib import Path
from vacation_router.gen_algo import GenePool
from vacation_router.parser import get_distance_graph, parse_user_inputs_to_df
from vacation_router.pathfinder import find_best_paths, get_path
from monty.serialization import dumpfn
from tqdm import tqdm

test_files_dir = Path("../test_files")
df = parse_user_inputs_to_df(
    str(test_files_dir / "Toronto2021.kml"), user_input_csv=str(test_files_dir / "interest_and_time.csv")
)
G = get_distance_graph(df)
# %%
least_time, parent = find_best_paths(G, 11)
# %%
# Remove the key were never set
least_time = {k: v for k, v in least_time.items() if v != float('inf')}

# %%
# Validate
def get_node_data_from_bitmask(G, bit_mask):
    return [G.nodes()[i] for i in range(len(G.nodes())) if bit_mask & (1 << i) != 0]

for (node, bit_mask), best_time in tqdm(least_time.items()):
    if best_time == float('inf'):
        continue
    node_data = get_node_data_from_bitmask(G, bit_mask)
    name_set = set([n['name'] for n in node_data])
    assert best_time, sum(df[df.name.isin(name_set)].time) / 60
least_time_dict = {k: v for k, v in least_time.items() if v != float('inf')}
parent_dict = {k: v for k, v in parent.items() if v != None}
dumpfn(least_time_dict, 'least_time.json', indent=2)
dumpfn(parent_dict, 'parent.json', indent=2)

# %%
best_end_for_bm = dict()
best_time_for_bm = defaultdict(lambda : float('inf'))
for (k, bm),v in tqdm(least_time.items()):
    if k < best_time_for_bm[bm]:
        best_end_for_bm[bm], best_time_for_bm[bm] = (k, bm), v
best_path_for_bm = {str(k): get_path(v, parent_dict, G) for k, v in best_end_for_bm.items()}
dumpfn(best_path_for_bm, 'best_path_for_bm.json', indent=2)
dumpfn(best_time_for_bm, 'best_time_for_bm.json', indent=2)

# %%
score_dict = {1<<k:v for k,v in enumerate(df.Total)}
def score_func(bm):
    score = 0
    while bm > 0:
        score += score_dict[bm & -bm]
        bm &= bm - 1
    return score

# %%
valid_bms = [int(bm) for bm in best_path_for_bm.keys() if len(best_path_for_bm[bm]) > 0]
genepool = GenePool(
    score_func=score_func,
    pop_size=100,
    n_masks=3,
    lam=2.0,
    valid_bitmasks=valid_bms,
    high_score=True,
    n_keep=10,
    max_attempts=1000,
    left_most=None
)

#%%
for i in tqdm(range(100)):
    print(genepool.population_fitness())
    genepool.step()
# %%
for i in genepool.population:
    print(i, genepool.fitness(i))

# %%
print(f"{14444:b}")

# %%
best_path_for_bm[str(14444)]
# %%

for bm in [14444, 639504, 3539330]:
    s = "Path: "
    for node in best_path_for_bm[str(bm)]:
        s += f"{node['name']} -> "
    print(s)


# %%
