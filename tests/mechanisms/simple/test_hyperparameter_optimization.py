import numpy as np
from apc_anonymizer.mechanisms.simple import hyperparameter_optimization


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
    assert np.all(np.abs(normalized.sum(axis=1) - 1.0) < 1e-10)
