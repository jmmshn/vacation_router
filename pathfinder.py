#%%
from collections import defaultdict, deque

from monty.serialization import dumpfn
from parser import *
from tqdm import tqdm
import logging

_logger = logging.getLogger(__name__)

df = parse_kml_to_df("./Toronto2021.kml", "./interest_and_time.csv")
df.to_csv('parsed_df.csv')
G = get_distance_graph(df)

# %%
def find_best_paths(G, max_time):
    """
    Find all the paths that can be taken under a given time.
    The underlying data structure is a deque. The state of system is represented by a node number and a bitmask.
    The bitmask represents the nodes that are visited, and the node number represents the final node in that set (history does not matter).
    So the state is represented by a tuple (node, bitmask).
    
    Args:
        G: The graph
        max_time: The maximum time allowed
    
    Returns:
        least_time: a dictionary of the least time to visit a set of nodes terminating at a given node
        parent: a dictionary of the parent state of a given state can be used to reconstruct the path
    """
    conn_dict = dict(G.adjacency())
# the least time should be keyed by the current node and the bit mask of visited nodes
    least_time = defaultdict(lambda: float('inf'))
    parent = defaultdict(lambda: None)

    queue = deque()
    for i in range(0, len(G.nodes())):
        new_time = G.nodes()[i]['time']
        if new_time < max_time:
            least_time[i, 1 << i] = new_time
            queue.append((i, 1 << i))

    while queue:
        curr_node, bit_mask = queue.popleft()
        if len(queue) % 10000 == 0:
            print(len(queue))
        
        _logger.debug(f"curr_node: {curr_node}, bit_mask {bit_mask:#06b}")
        
        if bit_mask == (1 << len(G.nodes())) - 1:
            break
    
        for nn in conn_dict[curr_node]:
        # if you have not visited the node yet
            if bit_mask & (1 << nn) == 0:
                new_time = least_time[curr_node, bit_mask] + G.nodes()[nn]['time'] + G.edges[curr_node, nn]['weight']
                new_bit_mask = bit_mask | (1 << nn)

                if new_time < least_time[nn, new_bit_mask] and new_time < max_time:
                    least_time[nn, new_bit_mask] = new_time
                    parent[nn, new_bit_mask] = (curr_node, bit_mask)
                    queue.append((nn, new_bit_mask))
    return least_time,parent
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
least_time_dict = {str(k): v for k, v in least_time.items() if v != float('inf')}
parent_dict = {str(k): v for k, v in parent.items() if v != None}
dumpfn(least_time_dict, 'least_time.json', indent=2)
dumpfn(parent_dict, 'parent.json', indent=2)

# %%
best_end_for_bm = dict()
best_time_for_bm = defaultdict(lambda : float('inf'))
for (k, bm),v in tqdm(least_time.items()):
    if k < best_time_for_bm[bm]:
        best_end_for_bm[bm], best_time_for_bm[bm] = (k, bm), v
# %%
def backtrack(k, d):
    """
    Go back from value->key to find the path.
    The order of the yield from statements means that the path is returned in the correct order.
    """
    if k not in d or d[k] is None:
        yield k
    else:
        yield from backtrack(d[k], d)
        yield k

def get_path(k, G):
    """
    Args:
        k: the key of the least time dictionary
    """
    path_ints = list(backtrack(k, parent))
    return [G.nodes[i] for i, _ in path_ints]
        
get_path((16, 65560), G)
# %%
best_path_for_bm = {k: get_path(v, G) for k, v in best_end_for_bm.items()}

# %%
dumpfn(best_path_for_bm, 'best_path_for_bm.json', indent=2)
dumpfn(best_end_for_bm, 'best_end_for_bm.json', indent=2)

# %%
