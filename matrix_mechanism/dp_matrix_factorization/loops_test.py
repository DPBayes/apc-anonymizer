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

"""Tests for loops."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from dp_matrix_factorization import constraint_builders
from dp_matrix_factorization import loops
from dp_matrix_factorization import matrix_constructors


def _make_prefixsum_s(dimensionality):
  return tf.constant(np.tril(np.ones(shape=(dimensionality, dimensionality))))


class LoopsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('random_true', True),
      ('random_false', False),
  )
  def test_streaming_results_respect_streaming_constraints(self, random_init):
    log_2_observations = 3
    s_matrix = _make_prefixsum_s(2**log_2_observations)
    if not random_init:
      initial_h = matrix_constructors.binary_tree_matrix(
          log_2_leaves=log_2_observations)
    else:
      initial_h = matrix_constructors.random_normal_binary_tree_structure(
          log_2_leaves=log_2_observations)
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=s_matrix,
        n_iters=10,
        optimizer=tf.keras.optimizers.SGD(0.1),
        streaming=True)
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    h_mask = matrix_constructors._compute_h_mask(binary_tree_h)
    streaming_vars_for_w = constraint_builders.compute_flat_vars_for_streaming(
        binary_tree_h)

    flat_w = np.reshape(solution['W'], [-1])
    for idx, value in enumerate(streaming_vars_for_w):
      if not value:
        self.assertEqual(flat_w[idx], 0.)

    disallowed_h_element_indicator = np.ones(shape=h_mask.shape) - h_mask
    self.assertEqual(np.max(disallowed_h_element_indicator * solution['H']), 0)

  @parameterized.named_parameters(
      ('streaming_false', False),
      ('streaming_true', True),
  )
  def test_factorizes_prefix_sum_matrix_single_iteration(self, streaming):
    log_2_observations = 1
    expected_s = _make_prefixsum_s(2**log_2_observations)
    initial_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=expected_s,
        n_iters=1,
        optimizer=tf.keras.optimizers.SGD(0),
        streaming=streaming)
    self.assertAllClose(solution['W'] @ solution['H'], expected_s)

  @parameterized.named_parameters(
      ('streaming_false', False),
      ('streaming_true', True),
  )
  def test_reduces_loss_while_factorizing_prefix_sum_small_matrix(
      self, streaming):
    log_2_observations = 1
    s_matrix = _make_prefixsum_s(2**log_2_observations)
    initial_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    expected_s = np.array([[1., 0.], [1., 1.]])
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=s_matrix,
        n_iters=10,
        optimizer=tf.keras.optimizers.SGD(0.001),
        streaming=streaming)
    self.assertAllClose(solution['W'] @ solution['H'], expected_s)
    self.assertLess(solution['loss_sequence'][-1], solution['loss_sequence'][0])

  @parameterized.named_parameters(
      ('streaming_false', False),
      ('streaming_true', True),
  )
  def test_reduces_loss_while_factorizing_prefix_sum_medium_matrix(
      self, streaming):
    log_2_observations = 5
    expected_s = _make_prefixsum_s(2**log_2_observations)
    initial_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=expected_s,
        n_iters=10,
        optimizer=tf.keras.optimizers.SGD(0.0001),
        streaming=streaming)
    self.assertAllClose(solution['W'] @ solution['H'], expected_s)
    self.assertLess(solution['loss_sequence'][-1], solution['loss_sequence'][0])


class FixedPointIterationsTest(tf.test.TestCase):

  def test_factorization_factorizes_target(self):
    log_2_observations = 1
    target_s = _make_prefixsum_s(2**log_2_observations).numpy()
    solution = loops.compute_h_fixed_point_iteration(
        s_matrix=target_s, rtol=1e-8)
    self.assertAllClose(solution['W'] @ solution['H'], target_s)
    # In seemingly a numerical accident, the 2x2 optimum is the golden ratio.
    self.assertAllClose(solution['loss'], 1.618034)
    # Assert that we took the positive factorization
    self.assertGreater(solution['W'][0][0], 0.)
    self.assertGreater(solution['W'][1][1], 0.)

  def test_lowering_tolerance_increases_solution_quality(self):
    log_2_observations = 2
    target_s = _make_prefixsum_s(2**log_2_observations).numpy()
    high_tol_solution = loops.compute_h_fixed_point_iteration(
        s_matrix=target_s, rtol=1e-1)
    low_tol_solution = loops.compute_h_fixed_point_iteration(
        s_matrix=target_s, rtol=1e-5)
    self.assertLess(low_tol_solution['loss'], high_tol_solution['loss'])


if __name__ == '__main__':
  tf.random.set_seed(2)
  np.random.seed(2)
  tf.test.main()
