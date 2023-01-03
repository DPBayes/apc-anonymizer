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
# Return to project root.
popd

echo "Requirements installed; beginning factorization"
# Move up a directory to ensure we can direct Python to run our binary as a
# submodule of factorize_prefix_sum.
pushd ..
  # Run simple settings for the binary.
  python -m dp_matrix_factorization.factorize_prefix_sum --strategy=identity --rtol=1e-5 --root_output_dir=/tmp/matfac --experiment_name=test_matrix_factorization
# Pop back down into project root.
popd
# Exit the virtualenv's Python environment.
deactivate
# Clean up tempdir.
rm -rf "$tmp_dir"
