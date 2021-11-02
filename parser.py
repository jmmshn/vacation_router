#%%
from collections import defaultdict
from networkx.algorithms.shortest_paths.unweighted import predecessor
from pykml import parser
import pandas as pd
import numpy as np
import networkx as nx
import heapq
from icecream import ic


def parse_kml_to_df(kml_file, interest_and_time):
    """
    Parses a KML file into a Pandas DataFrame.
    """
    data = []
    # Parse the KML file
    doc = parser.parse(kml_file).getroot()
    for e in doc.Document.Folder.Placemark:
        # Get the coordinates and convert to float
        coords = e.Point.coordinates.text.split(",")
        # Get the name
        name = e.name.text
        # Get the coordinates
        data.append([name, float(coords[0]), float(coords[1])])
    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=["name", "longitude", "latitude"])
    add_data = pd.read_csv(interest_and_time, index_col=["name"])
    df = df.join(add_data, how="inner", on="name")
    df.reset_index(inplace=True, drop=True)
    return df


def get_distance_graph(df, avg_speed=10):
    """
    Takes a Pandas DataFrame and returns a graph of the distances between
    each pair of points. The average speed of a bus is 10 m/s.
    Args:
        df: A Pandas DataFrame with the coordinates of the points.
        avg_speed: The average speed of the buses in km/h.
    Returns:
        A networkx graph.
    """
    # Create a graph
    G = nx.Graph()
    # Add the nodes
    G.add_nodes_from(df.index)
    # Add the edges
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            G.add_edge(
                df.index[i],
                df.index[j],
                weight=get_distance_meters(df.iloc[i], df.iloc[j]) / 1000 / avg_speed,
            )
    attrs = dict()
    for itr, row in df.iterrows():
        attrs[itr] = {"name": row["name"], "time": row.time / 60, "interest": row.Total}  # get time in hours
    # add attrs to nodes
    nx.set_node_attributes(G, attrs)
    return G


def get_distance_meters(row1, row2):
    """
    Returns the distance in meters between two rows of a Pandas DataFrame.
    """
    # Get the coordinates
    lon1 = row1.longitude
    lat1 = row1.latitude
    lon2 = row2.longitude
    lat2 = row2.latitude
    # Get the distance
    return get_distance_meters_from_coords(lon1, lat1, lon2, lat2)


def get_distance_meters_from_coords(lon1, lat1, lon2, lat2):
    """
    Returns the distance in meters between two coordinates.
    """
    # Get the distance
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d





def dijkstra_with_node_value(G, source):
    """
    Starting from a source point, returns the shortest path from the source to
    all other points in the graph. Also add the node value in the evaluation.
    Note: this can be done easier by using a DiGraph and adding the node times onto each arriving edge.
    But I thought think this is fun excercise to code up dijkstra's algorithm with additional operations with the node values.
    This might be useful for the future.
    Args:
        G: A networkx graph.
        source: The source node in the graph.
    """
    conn_dict = dict(G.adjacency())
    least_time = defaultdict(lambda: float("inf"))
    most_interest = defaultdict(lambda: float("inf"))
    def _get_time(node):
        return G.nodes[node]["time"]
    def _get_interest(node):
        return -G.nodes[node]["interest"] # negative since we want to maximize the interest
    pq = [(_get_interest(source), _get_time(source), source)]
    least_time[source] = _get_time(source)
    most_interest[source] = _get_interest(source)
    parents = dict()
    
    while pq:
        interest, time, node = heapq.heappop(pq)
        # print(cost, least_cost[node])
        if time > least_time[node]:
            continue
        # print(conn_dict[node])
        for neighbor, d_node in conn_dict[node].items():
            # print(neighbor, d_node["weight"])
            d_time = d_node["weight"] + _get_time(neighbor)
            new_time = time + d_time
            new_interest = interest + _get_interest(neighbor)
            # print(f"{node}->{neighbor}", new_cost, cost, d_node["weight"], _get_time(neighbor))
            if new_interest < most_interest[neighbor]:
                least_time[neighbor] = new_time
                most_interest[neighbor] = new_interest
                parents[neighbor] = node
                heapq.heappush(pq, (new_time, new_interest, neighbor))
    
    for node in G.nodes:
        ic(node, least_time[node], most_interest[node])
    
# %%
def bellman_ford(G, source):
    """
    Starting from a source point, returns the shortest path from the source to
    all other points in the graph.
    Args:
        G: A networkx graph.
        source: The source node in the graph.
    """
    most_interest = defaultdict(lambda: float("-inf"))
    least_time = defaultdict(lambda: float("inf"))
    visited = defaultdict(set) # set of visited nodes including the current node
    predecessor = defaultdict(lambda: None)

    def _get_time(node):
        return G.nodes[node]["time"]
    def _get_interest(node):
        return G.nodes[node]["interest"] 
    
    most_interest[source] = _get_interest(source)
    least_time[source] = _get_time(source)
    visited[source].add(source)
    # for i in range(len(G) - 1):
    for i in range(3):
        for u, v, d in G.edges(data=True):
            new_interest = most_interest[u] + _get_interest(v)
            new_time = least_time[u] + d["weight"] + _get_time(v)
            # print(f"{u}->{v}", new_interest, most_interest[v])
            if new_interest >= most_interest[v] and v not in visited[u] and d["weight"] < 0.2:
                most_interest[v] = new_interest
                least_time[v] = least_time[u] + d["weight"] + _get_time(v)
                predecessor[v] = u
                visited[v] = visited[u] | {v}
    return most_interest, least_time, visited
# most_interest, least_time, visited = bellman_ford(G, "Royal Ontario Museum")
# most_interest, least_time, visited = map(pd.Series, [most_interest, least_time, visited])
# df = pd.concat([most_interest, least_time, visited], axis=1)
# df.columns = ["interest", "time", "visited"]
# df1 = pd.DataFrame(df.visited.values.tolist()) \
#         .rename(columns = lambda x: 'visited{}'.format(x+1)) \
#         .fillna('-')
