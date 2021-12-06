import pytest
from pathlib import Path
from vacation_router.parser import get_distance_graph, parse_user_inputs_to_df


@pytest.fixture
def user_files():
    test_files_dir = Path(__file__).parent.parent / "test_files"
    return {
        "kml": str(test_files_dir / "Toronto2021.kml"),
        "loc_data": str(test_files_dir / "interest_and_time.csv"),
    }

@pytest.fixture
def user_input_df(user_files):
    df = parse_user_inputs_to_df(
        user_files["kml"], user_input_csv=user_files["loc_data"]
    )
    assert df.shape[0] == 24
    return df

@pytest.fixture
def ex_graph(user_input_df):
    return get_distance_graph(user_input_df)