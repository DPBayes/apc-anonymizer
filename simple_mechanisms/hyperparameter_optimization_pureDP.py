import optuna, jax, argparse

import numpy as np
import jax.numpy as jnp

from functools import partial
from infer import pure_dp_penalty, l2_penalty, distance_penalty, learn_with_sgd

## Set up the data

# following the waltti git
n_seats = 78
n_cats = 6
cat_edges = [5, 40, 50, 65, 72, 78]
#print(cat_edges)
categories = np.empty((n_seats+1, n_cats))
j = 0
for i in range(n_seats+1):
    if i > cat_edges[j]:
        j += 1
    categories[i] = np.eye(n_cats)[j]
#print(np.sum(categories, axis=0))

# compute distances between the categories across seats
distances = np.abs(np.arange(0,n_cats).reshape(-1, 1) - np.arange(0,n_cats).reshape(1, -1))
distance_matrix = categories @ distances


## optimize hyperparams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epsilon", type=float, default=1.0)
    #parser.add_argument("--delta", type=float, default=1e-6)
    parser.add_argument("--num_iters", type=int, default=int(1e6))
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--output_path", type=str, default="./")
    args = parser.parse_args()

    epsilon = args.epsilon

    # create a prototype penalties to initialize the task
    penalties = [
            (partial(pure_dp_penalty, eps=epsilon, n_seats=n_seats), 10.), 
            (l2_penalty, 0.001), 
            (partial(distance_penalty, distances=distance_matrix, n_seats=n_seats), 0.001)
            ]

    task = learn_with_sgd(n_seats, n_cats, categories, penalties)
    n_iters = args.num_iters

    def evaluate_results(logits):
        log_probs = jax.nn.log_softmax(logits.reshape(n_seats+1,-1), axis=1)
        # log-prob of reporting the correct category
        log_p_correct_category = log_probs[np.where(categories)]
        # DP guarantee
        row_wise_epsilon = jnp.abs(log_probs[1:] - log_probs[:-1]).max(1)

        return log_probs, log_p_correct_category, row_wise_epsilon

    def objective(trial):
        dp_weight = trial.suggest_float(name='dp weight', low=1e-3, high=1e1)
        l2_weight = trial.suggest_float(name='l2 weight', low=1e-4, high=1e0)
        dist_weight = trial.suggest_float(name='dist. weight', low=1e-5, high=1e-3)
        # replace the penalties in the task
        new_penalties = [(penalty, weight) for (penalty, old_weight), weight in zip(task.penalties, [dp_weight, l2_weight, dist_weight])]
        task.penalties = new_penalties
        # learn the parameters using SGD
        learned_logits = task.train(n_iters, init_seed=args.seed, silent=True)
        #
        log_probs, logp_correct, epsilons = evaluate_results(learned_logits)
        #
        logp_loss = np.linalg.norm(logp_correct)
        eps_loss = np.linalg.norm(epsilons-epsilon)
        furthest_cat_logp = log_probs[np.arange(n_seats+1), np.argmax(categories @ distances, axis=1)]
        dist_loss = np.sum(np.exp(furthest_cat_logp) > 1e-3)
        return logp_loss, eps_loss, dist_loss, np.linalg.norm(np.exp(furthest_cat_logp))

    study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
    study.optimize(objective, n_trials=args.num_trials)

if __name__ == "__main__":
    main()
