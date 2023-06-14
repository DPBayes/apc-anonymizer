"""Create initial dataframe for hyperparameter optimization."""

import numpy as np
import pandas as pd


def calculate_maximum_counts_from_minimum_counts(vehicle_model):
    return (
        list(map(lambda x: x - 1, vehicle_model["minimumCounts"].values()))[1:]
    ) + [vehicle_model["maximumCount"]]


def create_initial_dataframe(vehicle_model):
    """Create the initial matrix for hyperparameter optimization.

    For each row, the initial matrix has value 1.0 for one category and 0.0 for
    the others, according to the vehicle_model. The matrix describes the
    probabilities of selecting each category before any anonymization attempt
    has been made.
    """
    cat_names = vehicle_model["minimumCounts"].keys()
    cat_edges = calculate_maximum_counts_from_minimum_counts(vehicle_model)
    n_seats = max(cat_edges)
    n_cats = len(cat_names)
    categories = np.empty((n_seats + 1, n_cats))
    j = 0
    for i in range(n_seats + 1):
        if i > cat_edges[j]:
            j += 1
        categories[i] = np.eye(n_cats)[j]
    return pd.DataFrame(categories, columns=cat_names)
