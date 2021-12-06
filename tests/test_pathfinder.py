#!/usr/bin/env python
# %%
import pytest
from pathlib import Path
from vacation_router.pathfinder import find_best_paths

def test_find_best_paths(ex_graph):
    """
    Test that the function returns a list of the best paths
    """
    least_time, parent = find_best_paths(ex_graph, 4)
    assert len(parent) > 0

 
