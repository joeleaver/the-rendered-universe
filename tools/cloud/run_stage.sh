#!/bin/bash
# stage launcher — source-built block2 in the micromamba env;
# LD_LIBRARY_PATH points at the env's pip-MKL libs (same family the
# build linked against, so no kernel/core version schism).
export LD_LIBRARY_PATH=/home/ubuntu/mm_b2/lib
export B2_STACK_GB=80
cd /home/ubuntu/cloudrun || exit 1
exec /home/ubuntu/mm_b2/bin/python -u massgap.py --dmrg "$@"
