#!/bin/bash

set -Eeuo pipefail

JSON_SCHEMA_PATH='./src/apc_anonymizer/apc-anonymizer-schema.json'

npm_config_yes=true \
  npx \
  --package ajv-formats \
  --package ajv-cli \
  ajv \
  --spec=draft2020 \
  --strict=true \
  --strict-required=true \
  --allow-union-types \
  -c ajv-formats \
  compile \
  -s "${JSON_SCHEMA_PATH}" \
  -o \
  >/dev/null
