"""
This file gives an example of how one can pass the vehicle configuration to the DP-passenger count method

The main script will try to import a function called config from this module. This function gets optionally parameters from the main call, and
returns a numpy array. The returned array needs to be a binary array of size n_seats X n_categories, and each row contains a single 1 corresponding
to the category where the particular number of seats belongs to
"""
import argparse
import numpy as np
import pandas as pd


def config(unparsed_args):
    """
    The module can use unused arguments of the main file as an additional set of arguments. This requires that the keywords for this function are
    not used in the main script
    """
    config_parser = argparse.ArgumentParser("Example configuration")
    config_parser.add_argument('--n_seats', type=int, default=78)
    config_parser.add_argument('--n_cats', type=int, default=6)
    config_parser.add_argument('--cat_edges', nargs='+', type=int)
    config_parser.add_argument('--cat_names', nargs='+', type=str, default=None)
    args = config_parser.parse_args(unparsed_args)

    categories = np.empty((args.n_seats+1, args.n_cats))
    j = 0
    for i in range(args.n_seats+1):
        if i > args.cat_edges[j]:
            j += 1
        categories[i] = np.eye(args.n_cats)[j]
    if args.cat_names is None:
        return pd.DataFrame(categories)
    return pd.DataFrame(categories, columns=args.cat_names)
