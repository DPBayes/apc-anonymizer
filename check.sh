#!/bin/bash

set -Eeuo pipefail

poetry run black --check src tests &&
  poetry run ruff src tests &&
  poetry run pytest tests
