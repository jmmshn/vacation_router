#!/usr/bin/env python
# %%
import pytest
from pathlib import Path
from vacation_router.parser import get_distance_graph

def test_get_distance_graph(user_input_df):
    df = user_input_df
    graph = get_distance_graph(df)
    assert graph.number_of_nodes() == 24
    assert graph.number_of_edges() == 24 * 23 / 2
    # Make sure the all the nodes have 'time' and 'interest' attributes
    for node in graph.nodes:
        assert graph.nodes[node]["time"] > 0
        assert graph.nodes[node]["interest"] > 0
    # Make sure the all the edges have a weight that is positive
    for edge in graph.edges:
        assert graph.edges[edge]["weight"] > 0