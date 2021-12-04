#!/usr/bin/env python
# %%
import pytest
from pathlib import Path
from vacation_router.parser import parse_user_inputs_to_df, get_distance_graph


@pytest.fixture
def user_files():
    test_files_dir = Path(__file__).parent.parent / "test_files"
    return {
        "kml": str(test_files_dir / "Toronto2021.kml"),
        "loc_data": str(test_files_dir / "interest_and_time.csv"),
    }


def test_parse_user_inputs_to_df(user_files):
    df = parse_user_inputs_to_df(
        user_files["kml"], user_input_csv=user_files["loc_data"]
    )
    assert df.shape[0] == 24

def test_get_distance_graph(user_files):
    df = parse_user_inputs_to_df(
        user_files["kml"], user_input_csv=user_files["loc_data"]
    )
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