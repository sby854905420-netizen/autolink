#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"


python "$PROJECT_ROOT/Run_local/sql_generator.py" \
    --dataset_name "MMQA" \
