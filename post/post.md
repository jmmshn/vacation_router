# Planning Your Next Vacation Using Graph Theory and Genetic Algorithms

Planning a vacation is hard work, with so many different points of interest (POI) and limited time it is difficult to create an itinerary that simultaneously aims to hit as many points of interest as possible in an optimized fashion and does not make the vacation feel too stressful.
Additionally, since there are always more points of interest than you have time for, nailing down an itinerary means deciding which POI's to skip and living with the lingering doubt that the attaction you skipped would have provided a life-changing experience.
While the algorithm prefrsented here cannot garauntee that you won't miss out on any life-affirming moments in your travels, it will allow you to blame missed attractions on an internet stranger's silly algorithm which is the closest you can get to peace of of mind this situation

# The Human Inputs 

Before we can let the algorithm take over all the decision making responsibilities, we first have to gather data about the personal preferences of all the members of the party.
To do this we must first come up with an exhaustive list of attractions in a city.
Aside from the common list of attractions that can be found on any given travel blog and youtube video, a wonderful resource called Atlas Obscura provides a large cache of lesser-known attractions in each city.
Once such a list is compiled, the two most important pieces of information for each POI in this list are:

1. `dwell_time`: an estimate of the time spent at each POI 
2. `score`: some kind of positive-definite score assigned to each point of interest indicating the level of interest from the members.

# Estimating the travel times

[TODO] Google Map

Now that we have the 

# Parsing the data from the user inputs



# Pathfinding

Our path-finding algorithm address the question of 
"What is the least-time pathway through all combinations of nodes?"
To compute this we will perform as standerd minimum cost path-finding with the additional constraint that we can only directly compare paths that have visited the exact same set of nodes.
For book-keeping purposes (and for bit-twidling fun) we will use and integer bit mask to keep track of the which nodes have been visited.
The algorithm wil have use a queue that consisting of `(j, mask)` pairs where `j` is the node value to be process and `mask` is an integer where n-th bit from the right is set to `1` if and only if the n-th node has been visited. 
Note that the `j`-th bit of `mask` will always be `1` since we are assuming that `j` was the last visited node.
The complete algorithm is outlined below:

1. Initialize the queue with `(j, 1 << j)` for all `j` from `0` to `N-1`. 

2. Initialize a `least_time` dictionary with a default value of +\infty wich will be keyed by the `(j, mask)` pairs.

[TODO Pictures]

3. Pop a `(j, mask)` pair from the right end of the queue and check all the neighbors `nn` of node `j`.

4. If `nn` is a visited node in `mask` then we ignore this node since we don't want to visit the same place twice.  In bit language this means we will only perform futher checks if `mask & (1 << nn) == 0`.

5. If `nn` has not been visited before we will check if the new time beats the current best time to reach `nn` with same set of visited nodes: `least_time[nn, mask | (1 << nn)]`, if the new time is better we will updated the `least_time` dictionary with the new best time and add `(nn, mask | (1 << nn))` to the left end of the queue.

6. Whenever `least_time[nn, mask | (1 << nn)]` is updated we will also update a `parent[nn, mask | (1 << n)]` to point to `(j, mask)` so that we can trace back the path.

7. Repeat steps 3 to 5 until the queue is empty.

Since we are constrained by the amount of time in a given day, we should supply the algorithm with a `max_time` that limits the total time (dwell_time and travel_time) allowed for each path.


The final result of the algorithm are two dictionaries:

    1. `least_time`: a dictionary that records the least time to reach each node with the specific set of visited nodes.
    2. `parent`: a dictionary that records the previous node in the least-time path to reach each node with a specific set of visited nodes.

Assuming we don't care about where we start and end each day, the optimal path through any set of POI will just be the lowest time amoung all keys that share the same bitmask.
This can be computed quickly using the following code.
```python
best_end_for_bm = dict()
best_time_for_bm = defaultdict(lambda : math.inf)
for (k, bm),v in least_time.items():
    if k < best_time_for_bm[bm]:
        best_end_for_bm[bm], best_time_for_bm[bm] = (k, bm), v
```

The optimal path can be easily constructed by taking the `(k, bm)` pair for `best_end_for_bm[bm]` and tracing `parent[k, bm]` up until the bitmask only has a single bit.

# Finding the Best Combination With a Genetic Algorithm




