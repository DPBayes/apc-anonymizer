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

#!/bin/bash
# Assumed to be run from project root.
if [[ ! -f "requirements.txt" ]]; then
  echo "run.sh assumed to be run from matrix-factorization project root, including e.g. a requirements.txt. No requirements found."
  exit 1
fi

# Rename Python imports after wild transformation
find . -name "*.py" -exec sed -i  's/from dp_matrix_factorization.dp_matrix_factorization import/from dp_matrix_factorization import/g' {} +

# Initialize Python env.
initial_dir="$(pwd)"
# Create a temp dir in which to place the virtualenv
tmp_dir="$(mktemp -d -t dp-matfac-env-XXXX)"
# Enter tempdir; we will pop before leaving.
pushd "$tmp_dir"
  virtualenv -p python3 pip_env
  source pip_env/bin/activate
  pip install --upgrade pip
  pip install -r "$initial_dir/requirements.txt"
popd

echo "Requirements installed; beginning factorization"
# Move up a directory to ensure we can direct Python to run our binary as a
# submodule of factorize_prefix_sum.
pushd ..
  matfac_name="prefix_sum_factorization"
  root_dir="/tmp/matfac"
  # Run simple settings for the binary. We coupled the directory we write these
  # factorizations to to the directory we read from in the aggregator builder,
  # given the flag configuration below.
  python -m dp_matrix_factorization.factorize_prefix_sum --strategy=identity --rtol=1e-5 --log_2_observations=8 --root_output_dir="$root_dir" --experiment_name="$matfac_name"
  # Train for these 256 rounds. With these settings, on a 12-core machine, rounds
  # take ~ 6 sec on average. Metrics should log every 20 rounds by default. With
  # these toy settings, we may not see much progress.
  echo "Factorization computed; beginning training."
  python -m dp_matrix_factorization.dp_ftrl.run_stackoverflow --aggregator_method=opt_prefix_sum_matrix --total_rounds=256 --total_epochs=1 --clients_per_round=10 --root_output_dir="$root_dir" --experiment_name=stackoverflow_test_run --matrix_root_path="$root_dir/$matfac_name"
# Pop back down into project root.
popd
# Exit the virtualenv's Python environment.
deactivate
# Clean up tempdir.
rm -rf "$tmp_dir"
