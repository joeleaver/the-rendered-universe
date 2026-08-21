#!/bin/bash
# Source-build block2 on the VM — the proven local recipe (lean
# cmake configure, MKL from pip inside the env). The pip wheel is
# unusable here: its vendored MKL core mismatches its own pinned
# kernel package (undefined symbol at dispatch).
set -e
export DEBIAN_FRONTEND=noninteractive
sudo apt-get install -y -q build-essential cmake > /dev/null 2>&1 || true

MM=/home/ubuntu/bin/micromamba
ENV=/home/ubuntu/mm_b2
[ -x "$ENV/bin/python" ] || "$MM" create -y -q -p "$ENV" \
    -c conda-forge python=3.12 numpy scipy pillow pip
echo "=== env ready"
"$ENV/bin/pip" install -q mkl mkl-include mkl-devel pybind11 psutil
echo "=== mkl + pybind11 installed"

cd /home/ubuntu
"$ENV/bin/pip" download block2 --no-deps --no-binary block2 \
    -d /home/ubuntu/b2src -q
tar -xzf /home/ubuntu/b2src/block2-*.tar.gz -C /home/ubuntu/b2src
SRC=$(ls -d /home/ubuntu/b2src/block2-*/ | head -1)
echo "=== sdist at $SRC"

mkdir -p /home/ubuntu/b2build && cd /home/ubuntu/b2build
export PATH="$ENV/bin:$PATH"
export MKLROOT="$ENV"
CMAKE_POLICY_VERSION_MINIMUM=3.5 cmake "$SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wno-error=stringop-overflow -Wno-error=array-bounds -Wno-error=maybe-uninitialized" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE=/home/ubuntu/b2build/lib \
    -DUSE_MKL=ON -DBUILD_LIB=ON -DBUILD_EXE=OFF -DLARGE_BOND=ON \
    -DUSE_KSYMM=OFF -DUSE_COMPLEX=ON -DUSE_SG=ON -DUSE_SANY=OFF \
    -DUSE_SINGLE_PREC=OFF > /home/ubuntu/b2_cmake.log 2>&1
echo "=== configured"
cmake --build . -- -j30 > /home/ubuntu/b2_make.log 2>&1 \
    || { echo "=== BUILD_FAILED (see b2_make.log)"; exit 1; }
echo "=== compiled"

SP="$ENV/lib/python3.12/site-packages"
cp /home/ubuntu/b2build/lib/block2*.so "$SP/"
cp -r "$SRC/pyblock2" "$SP/"
echo "=== installed"

cd /home/ubuntu/cloudrun
LD_LIBRARY_PATH="$ENV/lib" "$ENV/bin/python" - <<'PYEOF'
import sys
sys.argv = ['x']
from massgap import b2_run
e = b2_run(4, 0, 0, 0, 300, tag='CT5')
print('free L=4 compute: %.6f (exact -22.627417)' % e)
PYEOF
echo "=== BUILD_OK"
