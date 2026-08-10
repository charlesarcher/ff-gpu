#!/usr/bin/env bash
# Self-test fixture: emits the known-good golden bytes for leg 65536,
# ignoring any harness-appended arguments. Exercises the harness's positive
# diff path (capture -> rc check -> timing normalization -> counts -> 3-way
# diff) on a fixture that MUST pass.
exec cat "$(dirname "${BASH_SOURCE[0]}")/../../goldens/out_ff_seg_65536.txt"
