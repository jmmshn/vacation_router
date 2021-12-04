#%%
from collections import defaultdict
from networkx.algorithms.shortest_paths.unweighted import predecessor
from pykml import parser
import pandas as pd
import numpy as np
import networkx as nx
import heapq
from icecream import ic


def parse_user_inputs_to_df(kml_file, user_input_csv):
    """
    Parses a KML file into a Pandas DataFrame.
    Args:
        kml_file: The path to the KML file.
        user_input_csv: A csv file with the interest score and time of each location
    Returns:
        A Pandas DataFrame with all the relevant information about the locations.
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
    add_data = pd.read_csv(user_input_csv, index_col=["name"])
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
            Default is 10 km/h which is roughly the average speed of public transit in major cities.
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
    # add data to nodes
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

