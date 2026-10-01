#!/usr/bin/env bash
# Full pipeline. Requires HCP Open Access credentials (see README section 3).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY tests/test_map_type_guard.py
$PY scripts/00_build_atlas.py --fetch
$PY scripts/01_fetch_effect_estimates.py
$PY scripts/02_fetch_rest.py
$PY scripts/03_fetch_anatomy.py
$PY scripts/10_interaction.py
$PY scripts/11_rest_control.py
$PY scripts/12_anatomy_control.py
$PY scripts/20_verify_expected.py
$PY scripts/30_make_figures.py
