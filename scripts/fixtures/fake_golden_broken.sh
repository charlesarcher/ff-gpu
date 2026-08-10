#!/usr/bin/env bash
# Self-test fixture: emits the golden bytes for leg 65536 with line 2
# (the sqrtLimit header) corrupted. The harness MUST detect this as a
# mismatch — proves the diff path actually catches drift.
exec sed '2s/.*/maxPrimeMapValue CORRUPTED sqrtLimit BROKEN/' \
  "$(dirname "${BASH_SOURCE[0]}")/../../goldens/out_ff_seg_65536.txt"
