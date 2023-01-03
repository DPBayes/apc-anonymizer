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

import asyncio
import collections
import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_matrix_factorization.dp_ftrl import training_loop

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _build_federated_averaging_process():
  return tff.learning.build_federated_averaging_process(
      _uncompiled_model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
      server_optimizer_fn=tf.keras.optimizers.SGD)


def _uncompiled_model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = _create_input_spec()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


def _batch_fn():
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64))
  return batch


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


def _read_from_csv(file_name):
  """Returns a list of fieldnames and a list of metrics from a given CSV."""
  with tf.io.gfile.GFile(file_name, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames
    csv_metrics = list(reader)
  return fieldnames, csv_metrics


class ExperimentRunnerTest(tf.test.TestCase):

  def test_raises_non_iterative_process(self):
    bad_iterative_process = _build_federated_averaging_process().next
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=[bad_iterative_process],
          client_datasets_fn=client_datasets_fn,
          validation_fn=validation_fn,
          total_epochs=1,
          total_rounds=10,
          experiment_name='non_iterative_process',
          root_output_dir=root_output_dir)

  def test_raises_non_callable_client_dataset(self):
    iterative_process = _build_federated_averaging_process()
    client_dataset = [[_batch_fn()]]

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_dataset,
          validation_fn=validation_fn,
          total_epochs=1,
          total_rounds=10,
          experiment_name='non_callable_client_dataset',
          root_output_dir=root_output_dir)

  def test_raises_non_callable_evaluate_fn(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    metrics_dict = {}
    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_datasets_fn,
          validation_fn=metrics_dict,
          total_epochs=1,
          total_rounds=10,
          experiment_name='non_callable_evaluate',
          root_output_dir=root_output_dir)

  def test_raises_non_str_output_dir(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    def validation_fn(model):
      del model
      return {}

    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_datasets_fn,
          validation_fn=validation_fn,
          total_epochs=1,
          total_rounds=10,
          experiment_name='non_str_output_dir',
          root_output_dir=1)

  def test_fedavg_training_decreases_loss(self):
    batch = _batch_fn()
    federated_data = [[batch]]
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    def validation_fn(model):
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True)
      model.assign_weights_to(keras_model)
      return {'loss': keras_model.evaluate(batch.x, batch.y)}

    initial_state = iterative_process.initialize()

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_epochs=1,
        total_rounds=10,
        experiment_name='fedavg_decreases_loss',
        root_output_dir=root_output_dir)
    self.assertLess(
        validation_fn(final_state.model)['loss'],
        validation_fn(initial_state.model)['loss'])

  def test_checkpoint_manager_saves_state(self):
    loop = asyncio.get_event_loop()
    experiment_name = 'checkpoint_manager_saves_state'
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    client_seed = 5

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_epochs=1,
        total_rounds=1,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir,
        clients_seed=client_seed,
    )

    program_state_manager = tff.program.FileProgramStateManager(
        os.path.join(root_output_dir, 'checkpoints', experiment_name))
    restored_state, restored_round = loop.run_until_complete(
        program_state_manager.load_latest((final_state, 0, 0)))

    self.assertEqual(restored_round, 0)

    keras_model = tff.simulation.models.mnist.create_keras_model(
        compile_model=True)
    restored_state[0].model.assign_weights_to(keras_model)
    restored_loss = keras_model.test_on_batch(federated_data[0][0].x,
                                              federated_data[0][0].y)
    final_state.model.assign_weights_to(keras_model)
    final_loss = keras_model.test_on_batch(federated_data[0][0].x,
                                           federated_data[0][0].y)
    self.assertEqual(final_loss, restored_loss)
    # We persist the seed we saw or generated to ensure that we can restore the
    # sampling procedure as appropriate.
    self.assertEqual(restored_state[1], client_seed)
    # We checkpointed at epoch 0, since the round was also 0.
    self.assertEqual(restored_state[2], 0)

  def test_fn_writes_metrics(self):
    experiment_name = 'test_metrics'
    iterative_process = _build_federated_averaging_process()
    batch = _batch_fn()
    federated_data = [[batch]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    def test_fn(model):
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True)
      model.assign_weights_to(keras_model)
      return {'loss': keras_model.evaluate(batch.x, batch.y)}

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_epochs=1,
        total_rounds=1,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir,
        rounds_per_eval=10,
        test_fn=test_fn)

    csv_file = os.path.join(root_output_dir, 'results', experiment_name,
                            'experiment.metrics.csv')
    fieldnames, metrics = _read_from_csv(csv_file)
    self.assertLen(metrics, 2)
    self.assertIn('test/loss', fieldnames)


class ClientIDShufflerTest(tf.test.TestCase):

  def test_raises_with_round_too_large(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer = training_loop.ClientIDShuffler(1, clients_data)
    epoch, round_num = 0, len(clients_data.client_ids)
    with self.assertRaises(ValueError):
      client_shuffer.sample_client_ids(round_num, epoch)

  def test_raises_epochs_going_backwards(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer = training_loop.ClientIDShuffler(1, clients_data)
    epoch, round_num = 0, len(clients_data.client_ids) - 1
    _, epoch = client_shuffer.sample_client_ids(round_num, epoch)
    self.assertEqual(epoch, 1)
    # We raise whenever were asked for an epoch smaller than one wever already
    # potentially started yielding samples from in this process.
    with self.assertRaises(ValueError):
      client_shuffer.sample_client_ids(round_num, 0)

  def test_samples_for_larger_epoch(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer = training_loop.ClientIDShuffler(
        1, clients_data, drop_remainder=True)
    epoch, round_num = 0, 0
    sample1, epoch = client_shuffer.sample_client_ids(round_num, epoch)
    self.assertEqual(epoch, 0)
    # A fresh 'first' sample in the second epoch.
    sample2, epoch = client_shuffer.sample_client_ids(
        len(clients_data.client_ids), 1)
    self.assertEqual(epoch, 1)
    self.assertNotEqual(sample1, sample2)

  def test_shuffling(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer = training_loop.ClientIDShuffler(1, clients_data)
    epoch, round_num = 0, 0
    total_epochs = 2
    epoch2clientid = [[] for _ in range(total_epochs)]
    while epoch < total_epochs:
      clients, new_epoch = client_shuffer.sample_client_ids(round_num, epoch)
      epoch2clientid[epoch].extend(clients)
      round_num += 1
      epoch = new_epoch
    self.assertCountEqual(epoch2clientid[0], epoch2clientid[1])

  def test_remainder(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer1 = training_loop.ClientIDShuffler(
        len(clients_data.client_ids) - 1, clients_data, drop_remainder=True)
    client_shuffer2 = training_loop.ClientIDShuffler(
        len(clients_data.client_ids) - 1, clients_data, drop_remainder=False)
    epoch1, epoch2, round_num = 0, 0, 0
    total_rounds = 2
    while round_num < total_rounds:
      clients1, epoch1 = client_shuffer1.sample_client_ids(round_num, epoch1)
      clients2, epoch2 = client_shuffer2.sample_client_ids(round_num, epoch2)
      round_num += 1
    self.assertLen(clients1, len(clients_data.client_ids) - 1)
    self.assertLen(clients2, 1)
    self.assertEqual(epoch1, 2)
    self.assertEqual(epoch2, 1)

  def test_deterministic_sequence_generated_with_seed(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer1 = training_loop.ClientIDShuffler(1, clients_data, seed=0)
    client_shuffer2 = training_loop.ClientIDShuffler(1, clients_data, seed=0)
    epoch1, epoch2, round_num = 0, 0, 0
    total_rounds = 10
    while round_num < total_rounds:
      clients1, epoch1 = client_shuffer1.sample_client_ids(round_num, epoch1)
      clients2, epoch2 = client_shuffer2.sample_client_ids(round_num, epoch2)
      round_num += 1
      self.assertEqual(clients1, clients2)
      self.assertEqual(epoch1, epoch2)


if __name__ == '__main__':
  tf.test.main()
