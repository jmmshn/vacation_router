#%%
from collections import deque, defaultdict
import itertools

from monty.serialization import dumpfn, loadfn
from parser import *
from tqdm import tqdm

df = pd.read_csv("parsed_df.csv")
G = get_distance_graph(df)
# %%
least_time_json = loadfn("least_time.json")
least_time_json = {eval(k): v for k, v in least_time_json.items()}
parent_json = loadfn("parent.json")
parent_json = {eval(k): tuple(v) for k, v in parent_json.items()}
least_time = defaultdict(lambda: float("inf"))
parent = defaultdict(lambda: None)
least_time.update(least_time_json)
parent.update(parent_json)

# %%
def get_node_data_from_bitmask(G, bit_mask):
    return [G.nodes()[i] for i in range(len(G.nodes())) if bit_mask & (1 << i) != 0]

for (node, bit_mask), best_time in tqdm(least_time_json.items()):
    if best_time == float('inf'):
        continue
    node_data = get_node_data_from_bitmask(G, bit_mask)
    name_set = set([n['name'] for n in node_data])
    assert best_time, sum(df[df.name.isin(name_set)].time) / 60

# %%
def get_sequence(parent, idx):
    seq = []
    while idx != None:
        seq.append(idx[0])
        idx = parent[idx]
    return seq[::-1]
# %%
best_time_for_bitmask = defaultdict(lambda: float('inf'))
best_seq_for_bitmask = defaultdict(lambda: None)
for (node, bit_mask), best_time in tqdm(least_time.items()):
    if best_time == float('inf'):
        continue
    if best_time < best_time_for_bitmask[bit_mask]:
        best_time_for_bitmask[bit_mask] = best_time
        best_seq_for_bitmask[bit_mask] = get_sequence(parent, (node, bit_mask))

    
# %%
def get_total_interest_from_bitmask(G, bit_mask):
    """
    Calculat total interest from bitmask
    """
    node_data = get_node_data_from_bitmask(G, bit_mask)
    return sum(df[df.name.isin([n['name'] for n in node_data])].Total)

ll_ = []
for k in tqdm(best_seq_for_bitmask.keys()):
    if best_time_for_bitmask[k] == float('inf'):
        continue
    total = get_total_interest_from_bitmask(G, k)
    ll_.append((k, f"{bit_mask:#030b}", total, best_time_for_bitmask[k], best_seq_for_bitmask[k]))
# %% 
df2 = pd.DataFrame(ll_, columns=['bit_mask', 'bit_mask_str', 'total', 'best_time', 'best_seq'] )
df2.sort_values(by=['total', 'best_time'], ascending=(False, True), inplace=True)
df2
mask2path = dict(zip(df2.bit_mask, df2.best_seq))

# %%
l_total_mask = [*zip(df2['total'], df2["bit_mask"])]
l_total_mask = [*filter(lambda x: x[0] > 0, l_total_mask)]
print(len(l_total_mask))

# %%
pair_sum = defaultdict(float)
pair_mask = defaultdict(lambda: 0)
for (t1, b1), (t2, b2) in tqdm(itertools.combinations(l_total_mask, 2), total=len(l_total_mask)*(len(l_total_mask)-1)//2):
    if b1 & b2:
        continue
    if t1 + t2 > pair_sum[b1|b2]:
        pair_sum[b1|b2] = t1 + t2
        pair_mask[b1|b2] = [b1, b2]
#%%
#%%
dd ={"pair_sum" : pair_sum, "pair_mask" : pair_mask}
dumpfn(dd, "pair_res.json", indent=2)

quad_total = float('-inf')
quad_list = None
# %%
for pm1, pm2 in tqdm(itertools.combinations(pair_sum.keys(), 2)):
    min_seent = min(min_seent, bin(pm1 & pm2).count('1'))
    if pm1 & pm2:
        continue
    if pair_sum[pm1] + pair_sum[pm2] > quad_total:
        quad_total = pair_sum[pm1] + pair_sum[pm2]
        quad_list = pair_mask[pm1] + pair_mask[pm2]


    
# # %%
# for bitmask, seq in best_seq_for_bitmask.items():
#     print(f"{bitmask}, {bitmask:#012b} ({best_time_for_bitmask[bitmask]}) -> {seq}")
# # %%
# path = [(10, 1024), (18, 263168), (12, 267264), (16, 332800), (3, 332808), (15, 365576)]
# df.iloc[[i[0] for i in path]]
# # %%
# # %%
# print(get_total_interest_from_bitmask(G, 2425818))

# # %%
# from monty.serialization import dumpfn

# # dumpfn({"least"})

# # %%
# %%

# %%

def get_locations_from_seq(df, seq):
    return [df[df.index == n]['name'].values[0] for n in seq]

print(get_locations_from_seq(df, [18, 2, 12, 9, 4, 3, 1, 7, 6]))
print('----')
print(get_locations_from_seq(df, [18, 2, 9, 4, 3, 1, 7, 11, 6]))