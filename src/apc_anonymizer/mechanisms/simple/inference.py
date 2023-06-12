import logging
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from jax import Array, value_and_grad
from jax.lax import fori_loop
from numpyro.optim import Adam
from tqdm.contrib.logging import logging_redirect_tqdm


def centered_softmax(logits, axis=1, pad_with=0.0):
    padded_logits = jnp.pad(logits, ((0, 0), (0, 1)), constant_values=pad_with)
    return jax.nn.softmax(padded_logits, axis=axis)


def centered_log_softmax(logits, axis=1, pad_with=0.0):
    padded_logits = jnp.pad(logits, ((0, 0), (0, 1)), constant_values=pad_with)
    return jax.nn.log_softmax(padded_logits, axis=axis)


use_centered_softmax = False

softmax = jax.nn.softmax
log_softmax = jax.nn.log_softmax

if use_centered_softmax:
    softmax = centered_softmax
    log_softmax = centered_log_softmax


def force_dp(categories, logits, epsilon_target, tau=1e-3, max_iter=1000):
    new_logits = np.zeros(logits.shape)
    logit_last = logits[0]
    new_logits[0] = logit_last
    for c in range(1, len(categories)):
        new_logits[c] = logits[c]
        ll_ratio = log_softmax(new_logits[c]) - log_softmax(logit_last)
        iter_nr = 0
        while jnp.abs(ll_ratio).max() > epsilon_target:
            if iter_nr >= max_iter:
                break
            new_logits[c] = jnp.where(
                ll_ratio < -epsilon_target, new_logits[c] + tau, new_logits[c]
            )
            new_logits[c] = jnp.where(
                ll_ratio > epsilon_target, new_logits[c] - tau, new_logits[c]
            )
            ll_ratio = log_softmax(new_logits[c]) - log_softmax(logit_last)
            iter_nr += 1
        logit_last = new_logits[c]
    return new_logits.flatten()


def distance_penalty(logits, distances):
    log_probs = log_softmax(logits, axis=1)
    return jnp.sum(log_probs * distances)


def l2_penalty(logits):
    return jnp.linalg.norm(logits)


def pure_dp_penalty(log_probs, eps):
    ll_ratio = jnp.abs(log_probs[:-1] - log_probs[1:])
    return jax.nn.relu(ll_ratio - (eps - 1e-5)).max()


def adp_penalty(log_probs, eps, delta_target):
    probs = jnp.exp(log_probs)
    delta_add = jnp.max(
        jnp.sum(
            jnp.clip(
                probs[:-1, :] - jnp.exp(eps) * probs[1:, :],
                a_min=0,
                a_max=None,
            ),
            1,
        )
    )
    delta_remove = jnp.max(
        jnp.sum(
            jnp.clip(
                probs[1:, :] - jnp.exp(eps) * probs[:-1, :],
                a_min=0,
                a_max=None,
            ),
            1,
        )
    )
    delta_total = jnp.max(jnp.array((delta_add, delta_remove)))
    return jax.nn.relu(delta_total - delta_target)


class LearnWithSGD:
    def __init__(self, categories: np.ndarray, total_penalty: Callable):
        """
        categories: a binary matrix with rows corresponding to one-hot encoded
                    true category
        penalties: an array containing (penalty, weight) tuples
        """
        self.categories = categories
        self.penalty_fn = total_penalty

    def loss(self, logits: Array) -> float:
        """
        logits: probability table logits
        """

        log_probs = log_softmax(logits, axis=1)

        # the probability of releasing the true category
        bce = np.sum(log_probs * self.categories)

        combined_loss = -1 * bce
        ## Add penalty terms to the combined loss
        combined_loss += self.penalty_fn(logits)

        return combined_loss

    def train(self, num_iters, optimizer=None, init_seed=123, silent=False):
        if optimizer is None:
            optimizer = Adam(1e-3)

        ## initialize
        logits0 = jax.random.normal(
            jax.random.PRNGKey(init_seed), shape=self.categories.shape
        )

        if use_centered_softmax:
            logits0 = jax.random.normal(
                jax.random.PRNGKey(init_seed),
                shape=(self.categories.shape[0], self.categories.shape[1] - 1),
            )

        optim_state = optimizer.init(logits0)

        def update_epoch(i, params):
            """
            This is a base function for the SGD optimization
            i: number of current iteration
            params: parameters for the optimizer (optimizer's state, loss of
                    last iteration)
            """
            optim_state, last_chunk_loss = params
            logits = optimizer.get_params(optim_state)
            loss_at_iter, grads = value_and_grad(partial(self.loss))(logits)
            optim_state = optimizer.update(grads, optim_state)
            return optim_state, loss_at_iter

        """
        Next we use the above function to optimize the parameters.
        We split the number of iterations into smaller chunks in order to get
        some prints for the evolution of the loss
        """

        if num_iters > 100:
            epoch_len = num_iters // 100  # this way we can get 100 prints
        else:
            epoch_len = 1

        iterator = tqdm.tqdm(range(num_iters // epoch_len), disable=silent)

        with logging_redirect_tqdm():
            for _epoch_nr in iterator:
                optim_state_new, loss_at_iter = fori_loop(
                    0, epoch_len, update_epoch, (optim_state, 0.0)
                )
                iterator.set_description(f"Loss {loss_at_iter.item():.2f}")
                if jnp.isnan(loss_at_iter):
                    # FIXME: is this an error?
                    logging.info("Nans!!!!!")
                    break
                optim_state = optim_state_new
            if silent:
                logging.info(f"Loss achieved with SGD: {loss_at_iter}")
        logits = optimizer.get_params(optim_state)
        return logits
