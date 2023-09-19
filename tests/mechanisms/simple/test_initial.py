import pandas as pd
from apc_anonymizer.mechanisms.simple import initial


def test_simplest_initial_dataframe():
    vehicleModel = {
        "outputFilename": "foo",
        "minimumCounts": {
            "EMPTY": 0,
            "FULL": 1,
        },
        "maximumCount": 1,
    }
    expected_initial_df = pd.DataFrame(
        data={"EMPTY": [1.0, 0.0], "FULL": [0.0, 1.0]}
    )
    initial_df = initial.create_initial_dataframe(vehicleModel)
    assert initial_df.equals(expected_initial_df)


def test_simple_initial_dataframe():
    vehicleModel = {
        "outputFilename": "foo",
        "minimumCounts": {
            "EMPTY": 0,
            "STANDING_ROOM_ONLY": 1,
            "FULL": 2,
        },
        "maximumCount": 3,
    }
    expected_initial_df = pd.DataFrame(
        data={
            "EMPTY": [1.0, 0.0, 0.0, 0.0],
            "STANDING_ROOM_ONLY": [0.0, 1.0, 0.0, 0.0],
            "FULL": [0.0, 0.0, 1.0, 1.0],
        }
    )
    initial_df = initial.create_initial_dataframe(vehicleModel)
    assert initial_df.equals(expected_initial_df)


def test_realistic_initial_dataframe():
    vehicleModel = {
        "outputFilename": "volvo-8908rle.csv",
        "minimumCounts": {
            "EMPTY": 0,
            "MANY_SEATS_AVAILABLE": 6,
            "FEW_SEATS_AVAILABLE": 36,
            "STANDING_ROOM_ONLY": 46,
            "CRUSHED_STANDING_ROOM_ONLY": 84,
            "FULL": 110,
        },
        "maximumCount": 126,
    }
    expected_initial_df = pd.DataFrame(
        data={
            "EMPTY": ([1.0] * (6 - 0)) + ([0.0] * (126 - 6 + 1)),
            "MANY_SEATS_AVAILABLE": (
                ([0.0] * 6) + ([1.0] * (36 - 6)) + ([0.0] * (126 - 36 + 1))
            ),
            "FEW_SEATS_AVAILABLE": (
                ([0.0] * 36) + ([1.0] * (46 - 36)) + ([0.0] * (126 - 46 + 1))
            ),
            "STANDING_ROOM_ONLY": (
                ([0.0] * 46) + ([1.0] * (84 - 46)) + ([0.0] * (126 - 84 + 1))
            ),
            "CRUSHED_STANDING_ROOM_ONLY": (
                ([0.0] * 84) + ([1.0] * (110 - 84)) + ([0.0] * (126 - 110 + 1))
            ),
            "FULL": (
                ([0.0] * 110)
                + ([1.0] * (126 - 110))
                + ([1.0] * (126 - 126 + 1))
            ),
        }
    )
    initial_df = initial.create_initial_dataframe(vehicleModel)
    assert initial_df.equals(expected_initial_df)
