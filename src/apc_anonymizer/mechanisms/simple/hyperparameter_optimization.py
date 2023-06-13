"""Optimize hyperparameters."""

import functools
import logging
import math
import multiprocessing
import os
import pathlib

import numpy as np
import numpyro
import optuna
import pandas as pd
import sqlalchemy

from apc_anonymizer.mechanisms.simple import database, inference, initial


def get_min_and_max_index_of_ones(column):
    return np.flatnonzero(column)[np.array([0, -1])]


def get_category_bounds(categories):
    min_and_max_idx_array = np.apply_along_axis(
        get_min_and_max_index_of_ones, 0, categories
    )
    return list(map(tuple, min_and_max_idx_array.transpose()))


def calculate_distance_matrix(categories):
    category_bounds = get_category_bounds(categories)
    distance_matrix = np.zeros(categories.shape)
    for i in range(categories.shape[0]):
        for j in range(categories.shape[1]):
            lower, upper = category_bounds[j]
            if i >= lower and i <= upper:
                distance_matrix[i, j] = 0
            else:
                distance_matrix[i, j] = min(
                    np.abs(i - lower), np.abs(i - upper)
                )
    return distance_matrix


def create_study(vehicle_model, db_name):
    db_url = sqlalchemy.engine.URL.create(
        drivername="postgresql+psycopg2",
        username="postgres",
        database=db_name,
    )
    storage = optuna.storages.RDBStorage(url=str(db_url))
    study = optuna.create_study(
        study_name=vehicle_model["outputFilenames"][0],
        direction="minimize",
        storage=storage,
        load_if_exists=True,
    )
    return study


def normalize_probabilities(probs, diff_eps=1e-10):
    normalized = probs
    row_sums = np.apply_along_axis(math.fsum, axis=1, arr=probs)
    if np.any(np.abs(row_sums - 1.0) >= diff_eps):
        logging.debug(
            f"The final probabilities matrix is not normalized. The row sums "
            f"are {row_sums}. Normalizing the final probabilities now."
        )
        normalized = probs * (1.0 / row_sums)[:, np.newaxis]
    return normalized


def run_inference(
    inference_config,
    vehicle_model,
    db_name,
    is_training=True,
    output_directory=None,
):
    categories_df = initial.create_initial_dataframe(vehicle_model)
    categories = categories_df.to_numpy()
    distance_matrix = calculate_distance_matrix(categories)
    normalized_distance_matrix = distance_matrix / distance_matrix.max()

    # Set DP limits.
    epsilon_target = inference_config["options"]["epsilon"]
    delta_target = inference_config["options"]["delta"]

    # Create prototype penalties to initialize the task.
    if delta_target == 0.0:
        dp_penalty_fn = functools.partial(
            inference.pure_dp_penalty, eps=epsilon_target
        )
    else:
        dp_penalty_fn = functools.partial(
            inference.adp_penalty,
            eps=epsilon_target,
            delta_target=delta_target,
        )

    n_iters = inference_config["options"][
        "numberOfIterationsPerHyperparameterTrial"
    ]

    study = create_study(vehicle_model, db_name)

    if is_training:

        def objective(trial):
            dp_weight = trial.suggest_float(
                name="dp weight", low=1e2, high=1e4
            )
            dist_weight = trial.suggest_float(
                name="dist. weight", low=1e-6, high=1e-2
            )

            # Replace the penalties in the task.
            def total_penalty(qs):
                log_probs = inference.log_softmax(qs, axis=1)
                return dp_weight * dp_penalty_fn(
                    log_probs
                ) + dist_weight * inference.distance_penalty(
                    qs, normalized_distance_matrix
                )

            task = inference.LearnWithSGD(categories, total_penalty)

            # Learn the parameters using SGD.
            learned_logits = task.train(
                n_iters,
                init_seed=0,
                silent=True,
                optimizer=numpyro.optim.Adam(1e-3),
            )

            learned_log_probs = inference.log_softmax(learned_logits, axis=1)
            # Evaluate the DP guarantee.
            # can be either in terms of epsilon or delta, but does not really
            # matter
            # dp_excess = dp_penalty_fn(learned_logits)
            dp_excess = dp_penalty_fn(learned_log_probs)
            dp_loss = 0.0
            if dp_excess > 0.0:
                dp_loss = np.inf

            # Evaluate the average probability of releasing distant category.
            distance_loss = np.mean(
                inference.softmax(learned_logits) * (distance_matrix > 1)
            )

            return dp_loss + distance_loss

        n_trials = inference_config["options"][
            "numberOfHyperparameterTrialsPerProcess"
        ]
        study.optimize(objective, n_trials=n_trials)
    else:
        optimal_hyperparameters = study.best_params
        dp_weight = optimal_hyperparameters["dp weight"]
        dist_weight = optimal_hyperparameters["dist. weight"]

        def total_penalty(qs):
            log_probs = inference.log_softmax(qs, axis=1)
            return dp_weight * dp_penalty_fn(
                log_probs
            ) + dist_weight * inference.distance_penalty(
                qs, normalized_distance_matrix
            )

        task = inference.LearnWithSGD(categories, total_penalty)

        # learn the parameters using SGD
        learned_logits = task.train(
            n_iters,
            init_seed=0,
            silent=True,
            optimizer=numpyro.optim.Adam(1e-3),
        )

        ## Empirical check for DPness

        final_ps = inference.softmax(learned_logits)
        final_ps = normalize_probabilities(final_ps)
        final_log_ps = np.log(final_ps)

        if delta_target == 0:
            if dp_penalty_fn(final_log_ps) > 0.0:
                logging.warning(
                    f"Current configuration violates the DP requirement with "
                    f"epsilon "
                    f"{epsilon_target + dp_penalty_fn(final_log_ps)}, when "
                    f"the target epsilon was set to {epsilon_target}"
                )

        if delta_target > 0:
            if dp_penalty_fn(final_log_ps) > 0.0:
                logging.warning(
                    f"Current configuration violates the DP requirement with "
                    f"delta {delta_target + dp_penalty_fn(final_log_ps)}, "
                    f"when the target delta was set to {delta_target}"
                )

        if np.any(final_ps < 0):
            logging.warning(
                "The final probabilities matrix contains negative values. "
                "Writing it anyway."
            )

        ## store the learned probability table into a csv

        directory = pathlib.Path(output_directory)
        prob_df = pd.DataFrame(final_ps, columns=categories_df.columns)
        for output_filename in vehicle_model["outputFilenames"]:
            prob_df.to_csv(
                directory / output_filename,
                index_label="passenger_count",
                float_format="%.16g",
            )


def get_parallel_process_count(inference_config):
    n_processes = inference_config["options"]["numberOfProcesses"]
    if n_processes == "one-per-core":
        n_processes = len(os.sched_getaffinity(0))
    return n_processes


def run_hyperparameter_optimization_in_parallel(
    inference_config, vehicle_model, db_name
):
    # Create the tables before the child processes are created.
    create_study(vehicle_model, db_name)
    n_processes = get_parallel_process_count(inference_config)
    processes = []
    logging.info(f"Running optimization in {n_processes} processes")
    for _i in range(n_processes):
        p = multiprocessing.Process(
            target=run_inference,
            args=(inference_config, vehicle_model, db_name),
        )
        processes.append(p)
        p.start()
    [p.join() for p in processes]
    for p in processes:
        if p.exitcode != 0:
            raise RuntimeError(
                f"Hyperparameter optimization process with PID {p.pid} exited "
                f"abnormally."
            )


def run_inference_in_parallel(
    inference_config, vehicle_model, db_name, output_directory
):
    run_hyperparameter_optimization_in_parallel(
        inference_config, vehicle_model, db_name
    )
    run_inference(
        inference_config,
        vehicle_model,
        db_name,
        is_training=False,
        output_directory=output_directory,
    )


def run_inference_for_all_vehicle_models(config):
    logging.info("Prepare database")
    db_name = "hyperparameter_opt"
    database.prepare_database(db_name)
    for vehicle_model in config["vehicleModels"]:
        logging.info(
            f"Run inference for vehicle model "
            f"{vehicle_model['outputFilenames']}"
        )
        run_inference_in_parallel(
            config["inference"],
            vehicle_model,
            db_name,
            output_directory=config["outputDirectory"],
        )
    logging.info("Shutdown database")
    database.close_database()
