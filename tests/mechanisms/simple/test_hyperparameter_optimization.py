import numpy as np
import pytest
from apc_anonymizer.mechanisms.simple import (
    hyperparameter_optimization,
    initial,
)


def test_get_min_and_max_index_of_ones():
    column = np.array([0, 0, 1, 1, 1, 0])
    expected = np.array([2, 4])
    try:
        np.testing.assert_array_equal(
            hyperparameter_optimization.get_min_and_max_index_of_ones(column),
            expected,
            strict=True,
        )
    except AssertionError as e:
        pytest.fail(f"Getting min and max index of ones failed: {e}")


def test_get_min_and_max_index_of_ones_single():
    column = np.array([0, 0, 1, 0])
    expected = np.array([2, 2])
    try:
        np.testing.assert_array_equal(
            hyperparameter_optimization.get_min_and_max_index_of_ones(column),
            expected,
            strict=True,
        )
    except AssertionError as e:
        pytest.fail(
            f"Getting min and max index of just a single one failed: {e}"
        )


def test_get_category_bounds():
    vehicle_model = {
        "outputFilenames": ["foo.csv", "bar.csv"],
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
    expected = [(0, 5), (6, 35), (36, 45), (46, 83), (84, 109), (110, 126)]
    categories = initial.create_initial_dataframe(vehicle_model).to_numpy()
    assert (
        hyperparameter_optimization.get_category_bounds(categories) == expected
    )


def test_normalize_probabilities():
    probs = np.array(
        [
            [0.1234, 0.2345, 0.3456],
            [0.4567, 0.5678, 0.6789],
            [0.7890, 0.8901, 0.9012],
            [0.9123, 0.9234, 0.9345],
            [0.9456, 0.9567, 0.9678],
        ]
    )
    normalized = hyperparameter_optimization.normalize_probabilities(probs)
    try:
        np.testing.assert_allclose(normalized.sum(axis=1), 1.0, rtol=1e-10)
    except AssertionError as e:
        pytest.fail(f"Normalizing probabilities failed: {e}")


def test_calculate_distance_matrix_simple():
    vehicle_model = {
        "outputFilenames": ["foo.csv", "bar.csv"],
        "minimumCounts": {
            "EMPTY": 0,
            "FEW_SEATS_AVAILABLE": 3,
            "FULL": 4,
        },
        "maximumCount": 9,
    }
    expected_distance_matrix = np.array(
        [
            [0, 3, 4],
            [0, 2, 3],
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
            [3, 2, 0],
            [4, 3, 0],
            [5, 4, 0],
            [6, 5, 0],
            [7, 6, 0],
        ]
    )
    categories_df = initial.create_initial_dataframe(vehicle_model)
    categories = categories_df.to_numpy()
    distance_matrix = hyperparameter_optimization.calculate_distance_matrix(
        categories
    )
    try:
        np.testing.assert_allclose(distance_matrix, expected_distance_matrix)
    except AssertionError as e:
        pytest.fail(f"Calculating distance matrix for simple case failed: {e}")
