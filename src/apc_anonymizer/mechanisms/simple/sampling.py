import math

import numpy as np
import pandas as pd


def clamp(n, smallest, largest):
    """Force a value between the smallest and largest allowed value."""
    return max(smallest, min(n, largest))


def create_sampler(csv_path_or_buffer):
    """Create a sample function for choosing the occupancy status value.

    Run create_sampler once for each vehicle model that you have a profile for.
    """
    probabilities_df = pd.read_csv(
        csv_path_or_buffer, index_col="passenger_count"
    )
    min_count = probabilities_df.index[0]
    max_count = probabilities_df.index[-1]

    # Let's normalize the probabilities. They should already be normalized but
    # due to the floating-point serialization into CSV and deserialization out
    # of CSV there might be a small difference. np.random.choice is really
    # picky about normalization.
    probabilities = probabilities_df.values
    row_sums = np.apply_along_axis(math.fsum, axis=1, arr=probabilities)
    normalized = probabilities * (1.0 / row_sums)[:, np.newaxis]
    categories = probabilities_df.columns

    def sample(passenger_count):
        """Sample from the probabilities to produce an occupancy status.

        When you receive new passenger count data and need to update the
        published occupancy status, e.g. after every stop with changes in
        passenger count, call this function with the current passenger_count
        and publish the result.
        """
        clamped_count = clamp(passenger_count, min_count, max_count)
        probabilities_given_passenger_count = normalized[clamped_count, :]
        return np.random.choice(
            categories, p=probabilities_given_passenger_count
        )

    return sample
