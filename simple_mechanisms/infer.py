import logging

import jax, tqdm
import jax.numpy as jnp
import numpy as np

from jax import grad, jacobian, hessian, value_and_grad
from jax.nn import softmax, log_softmax
from jax.lax import fori_loop

from scipy.optimize import minimize

from numpyro.optim import Adam

from functools import partial

from jaxlib.xla_extension import DeviceArray

from collections.abc import Callable

from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger("joint_logger")

def distance_penalty(qs, distances, n_seats):
    logits = qs.reshape(n_seats+1, -1)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    return jnp.sum(log_probs * distances)

def l2_penalty(qs):
    return jnp.linalg.norm(qs)

def pure_dp_penalty(qs, eps, n_seats):
    logits = qs.reshape(n_seats+1, -1)
    log_probs = jax.nn.log_softmax(logits, axis=1)
    ll_ratio = jnp.abs(log_probs[:-1] - log_probs[1:])
    return jax.nn.relu(ll_ratio - (eps-1e-5)).max()
    #return jax.nn.relu(ll_ratio - (eps-1e-5)).sum()

def adp_penalty(qs, eps, delta_target, n_seats):
    logits = qs.reshape(n_seats+1, -1)
    probs = jnp.exp(jax.nn.log_softmax(logits, axis=1))
    delta_add = jnp.max(jnp.sum(jnp.clip(probs[:-1,:] - jnp.exp(eps)*probs[1:,:], a_min=0, a_max=None),1))
    delta_remove = jnp.max(jnp.sum(jnp.clip(probs[1:,:] - jnp.exp(eps)*probs[:-1,:], a_min=0, a_max=None),1))
    delta_total = jnp.max( jnp.array((delta_add,delta_remove)))
    return jax.nn.relu(delta_total - delta_target)

class learn_with_sgd:
    def __init__(self, n_seats: int, n_cats: int, categories: np.ndarray, total_penalty: Callable):
        """
        n_seats: total number of seats in a bus
        n_cats: number of categories to release the status from
        categories: a binary matrix with rows corresponding to one-hot encoded true category
        penalties: an array containing (penalty, weight) tuples
        """
        self.n_seats = n_seats
        self.n_cats = n_cats
        self.categories = categories
        self.penalty_fn = total_penalty
        
    def loss(self, qs: DeviceArray) -> float:
        """
        qs: probability table logits in long form (flattened)
        """

        logits = qs.reshape(self.n_seats+1, -1)
        log_probs = jax.nn.log_softmax(logits, axis=1)

        # the probability of releasing the true category 
        bce = np.sum(log_probs * self.categories)

        combined_loss = -1 * bce
        ## Add penalty terms to the combined loss
        combined_loss += self.penalty_fn(qs)

        return combined_loss


    def train(self, num_iters, optimizer=None, init_seed=123, silent=False):

        if optimizer is None:
            optimizer = Adam(1e-3)

        # initialize 
        qs0 = jax.random.normal(
                            jax.random.PRNGKey(init_seed), 
                            shape=(int((self.n_seats + 1) * self.n_cats),)
                        )

        optim_state = optimizer.init(qs0)
        
        def update_epoch(i, params):
            """
            This is a base function for the SGD optimization
            i: number of current iteration
            params: parameters for the optimizer (optimizer's state, loss of last iteration)
            """
            optim_state, last_chunk_loss = params
            qs = optimizer.get_params(optim_state)
            loss_at_iter, grads = value_and_grad(partial(self.loss))(qs)
            optim_state = optimizer.update(grads, optim_state)
            return optim_state, loss_at_iter

        """
        Next we use the above function to optimize the parameters.
        We split the number of iterations into smaller chunks in order to get some prints for the evolution of the loss
        """
        
        if num_iters > 100:
            epoch_len = num_iters // 100 # this way we can get 100 prints
        else: 
            epoch_len = 1

        iterator = tqdm.tqdm(range(num_iters // epoch_len), disable=silent)

        with logging_redirect_tqdm():
            for epoch_nr in iterator:
                optim_state_new, loss_at_iter = fori_loop(0, epoch_len, update_epoch, (optim_state, 0.0))
                iterator.set_description(f"Loss {loss_at_iter.item():.2f}")
                if jnp.isnan(loss_at_iter):
                    logger.info("Nans!!!!!")
                    break
                optim_state = optim_state_new
            if silent:
                logger.info(f"Loss achieved with SGD: {loss_at_iter}")
        logits = optimizer.get_params(optim_state)
        return logits
