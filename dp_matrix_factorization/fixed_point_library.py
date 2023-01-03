# coding=utf-8
# Copyright 2022 The Dp Matrix Factorization Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for computing lagrange multiplier settings for DP-MatFac."""
from typing import Optional, Tuple

from absl import logging

import jax
from jax import numpy as jnp
from jax.config import config

# With large matrices, the extra precision afforded by performing all
# computations in float64 is critical.
config.update('jax_enable_x64', True)


@jax.jit
def diagonalize_and_take_jax_matrix_sqrt(matrix,
                                         min_eigenval = 0.0
                                        ):
  """Matrix square root for positive-definite, Hermitian matrices."""
  evals, evecs = jnp.linalg.eigh(matrix)
  eval_sqrt = jnp.maximum(evals, min_eigenval)**0.5
  sqrt = evecs @ jnp.diag(eval_sqrt) @ evecs.T
  return sqrt


def compute_phi_fixed_point(
    target,
    rtol = 1e-5,
    max_iterations = None,
):
  """Computes fixed point of phi as defined in Theorem 3.2.

  Args:
    target: Rank-2 array (IE, matrix) playing the role of S*S in the definition
      of phi. Assumed to be Hermitian and positive-definite.
    rtol: Relative tolerance to use for computing fixed point; IE, a point x
      will be returned when norm(phi(x) - x) < rtol * norm(x).
    max_iterations: Optional int specifying the maximum number of iterations
      used to compute the fixed point of phi. If `None`, this function will loop
      until `rtol` tolerance is achieved. For extremely low values or `rtol` (or
      large `target` matrices), issues of numerical precision may make this
      condition extremely difficult to achieve.

  Returns:
    A tuple (v, n_iters, rel_norm_diff), where:
      * v is an approximate fixed-point to phi
      * n_iters is the number of iterations used to compute this fixed point
      * rel_norm_diff is the relative norm diff of the final iteration.

    Either `rtol` is achieved or `max_iterations` has been hit when we return.
  """

  if len(target.shape) != 2:
    raise ValueError('Expected `target` argument to be a rank-2 ndarray (IE, a '
                     f'matrix); found an array of rank {len(target.shape)}')
  if target.shape[0] != target.shape[1]:
    raise ValueError('Expected target to be a square matrix; found matrix of '
                     f'shape {target.shape}')
  if not jnp.all(jnp.isfinite(target)):
    raise ValueError('Cannot compute fixed-point for matrix with nan entries.')

  target_sqrt = diagonalize_and_take_jax_matrix_sqrt(target)
  # TODO: Consider different initializations and document how
  # initialization was chosen.
  v = jnp.diag(target_sqrt)
  target = target.astype(v.dtype)
  n_iters = 0

  def continue_loop(iteration):
    if max_iterations is None:
      return True
    return iteration < max_iterations

  while continue_loop(n_iters):
    n_iters += 1
    diag = jnp.diag(v)
    diag_sqrt = diag**0.5
    new_v = jnp.diag(
        diagonalize_and_take_jax_matrix_sqrt(diag_sqrt @ target @ diag_sqrt))
    assert jnp.all(new_v > 0)
    norm_diff = jnp.linalg.norm(new_v - v)
    rel_norm_diff = norm_diff / jnp.linalg.norm(v)
    logging.info('Norm diff: %s', norm_diff)
    logging.info('Relative norm diff: %s', rel_norm_diff)
    if rel_norm_diff < rtol:
      return new_v, n_iters, rel_norm_diff
    v = new_v

  return v, n_iters, rel_norm_diff
