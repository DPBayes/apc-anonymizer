import jax, argparse, importlib, os
import jax.numpy as jnp
import numpy as np
import pandas as pd

from numpyro.optim import Adam, SGD
from functools import partial

import optuna
from optuna.storages import RDBStorage

import sqlalchemy
from sqlalchemy.engine import URL

from utils import load_configuration, ConfigParsingException
from infer import adp_penalty, pure_dp_penalty, l2_penalty, distance_penalty, learn_with_sgd, softmax, log_softmax

np.random.seed(123)
np.set_printoptions(suppress=True)

################################################################################

def inference(args, unknown_args):
    ## load the configuration
    if not args.config_module_path[-3:] == ".py":
        raise ConfigParsingException("The config path must be a python module (i.e. end in '.py')")

    config = load_configuration(args.config_module_path)
    categories_df = config(unknown_args)
    categories = categories_df.values

    ## compute the distance between categories in the given configuration

    category_bounds = []
    for j in range(categories.shape[1]):
        indices_for_j = np.where(categories[:, j] == 1.0)[0]
        l = min(indices_for_j)
        u = max(indices_for_j)
        category_bounds.append((l, u))

    distance_matrix = np.zeros(categories.shape)
    for i in range(categories.shape[0]):
        for j in range(categories.shape[1]):
            l, u = category_bounds[j]
            if i>=l and i<=u:
                distance_matrix[i, j] = 0.
            else:
                distance_matrix[i, j] = min(np.abs(i-l), np.abs(i-u))
    normalized_distance_matrix = distance_matrix / distance_matrix.max()

    ## set DP limits

    epsilon_target = args.epsilon
    delta_target = args.delta

    assert epsilon_target > 0 and 0 <= delta_target <= 1, f'Invalid privacy params: eps={epsilon_target}, delta={delta_target}!'

    # create a prototype penalties to initialize the task
    if delta_target == 0:
        dp_penalty_fn = partial(pure_dp_penalty, eps=epsilon_target)
    else:
        dp_penalty_fn = partial(adp_penalty, eps=epsilon_target, delta_target=delta_target)

    n_iters = args.num_iters

    def objective(trial):
        dp_weight = trial.suggest_float(name='dp weight', low=1e2, high=1e4)
        dist_weight = trial.suggest_float(name='dist. weight', low=1e-6, high=1e-2)

        # replace the penalties in the task
        total_penalty = lambda qs: dp_weight * dp_penalty_fn(qs) + dist_weight * distance_penalty(qs, normalized_distance_matrix)
        task = learn_with_sgd(categories, total_penalty)

        # learn the parameters using SGD
        learned_logits = task.train(n_iters, init_seed=0, silent=True, optimizer=Adam(1e-3))

        # evaluate the DP guarantee
        dp_excess = dp_penalty_fn(learned_logits) # can be either in terms of epsilon or delta, but does not really matter
        if dp_excess > 0.:
            dp_loss = np.inf
        else:
            dp_loss = 0.0

        # evaluate the average probability of releasing distant category
        distance_loss = np.mean(softmax(learned_logits) * (distance_matrix > 1))

        return dp_loss + distance_loss


    # save results using postgresql database
    # SET THESE to point to some actual database
    db_url = URL.create(
            "postgresql",
            host="localhost",
            #database="hyperparameter_test_040122",)
            database="hyperparameter_opt",)

    storage = RDBStorage(url=str(db_url))

    #study = optuna.create_study(study_name=f"testing_again_040122_eps{epsilon_target}_delta{delta_target}", direction="minimize",
    study = optuna.create_study(study_name=f"{args.config_name}_eps{epsilon_target}_delta{delta_target}", direction="minimize",
                                storage=storage, load_if_exists=True)

    if args.train_hyperparameters:
        study.optimize(objective, n_trials=args.num_trials)

    else:
        optimal_hyperparameters = study.best_params
        dp_weight = optimal_hyperparameters["dp weight"]
        dist_weight = optimal_hyperparameters["dist. weight"]

        total_penalty = lambda qs: dp_weight * dp_penalty_fn(qs) + dist_weight * distance_penalty(qs, normalized_distance_matrix)
        task = learn_with_sgd(categories, total_penalty)

        # learn the parameters using SGD
        learned_logits = task.train(n_iters, init_seed=0, silent=True, optimizer=Adam(1e-3))

        ## Empirical check for DPness

        final_ps = softmax(learned_logits)
        logps = np.log(final_ps)
        if delta_target == 0:
            max_abs_logdiff_for_count = np.max(np.array([np.abs(logps[i] - logps[i+1]) for i in range(0, len(n_seats))]), axis=1)
            print(max_abs_logdiff_for_count)
            print(max_abs_logdiff_for_count.max())
            print(np.max(max_abs_logdiff_for_count) < epsilon_target)

        if delta_target > 0:
            print(dp_penalty_fn(learned_logits) / delta_target)
            print(dp_penalty_fn(learned_logits))

        ## store the learned probability table into a csv

        prob_df = pd.DataFrame(final_ps, columns=categories_df.columns)
        prob_df.to_csv(args.output_path + f"{args.config_name}_prob_table_eps{epsilon_target}_delta_{delta_target}.csv")


################################################################################

def sampling(args, unknown_args):

    epsilon_target = args.epsilon
    delta_target = args.delta

    ## read prob table
    prob_df = pd.read_csv(args.output_path + f"{args.config_name}_prob_table_eps{epsilon_target}_delta_{delta_target}.csv", index_col=0)
    
    ## sample according to row corresponding to args.state
    probs = prob_df.loc[args.state].values
    # normalize (shouldn't be necessary though)
    if probs.sum != 1.:
        probs = probs / probs.sum()
    draw = np.random.choice(prob_df.columns, p=probs)
    print(draw)
    return draw


#################################################################################
def main():
    parser = argparse.ArgumentParser()
    # common arguments
    parser.add_argument("config_module_path", type=str, help="Path to the vehicle configuration file")
    parser.add_argument("config_name", type=str, help="Name of your configuration")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--output_path", type=str, default="./")

    subparsers = parser.add_subparsers()

    # inference mode arguments
    parser_infer = subparsers.add_parser('infer')
    parser_infer.add_argument("--num_iters", type=int, default=int(1e7))
    parser_infer.add_argument("--train_hyperparameters", action='store_true')
    parser_infer.add_argument("--num_trials", type=int, default=50)
    parser_infer.set_defaults(func=inference)

    # sampling mode arguments
    parser_sample = subparsers.add_parser('sample')
    parser_sample.add_argument("state", type=int)
    parser_sample.set_defaults(func=sampling)

    args, unknown_args = parser.parse_known_args()

    args.func(args, unknown_args)

if __name__ == "__main__":
    main()
