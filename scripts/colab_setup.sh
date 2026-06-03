#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/colab_setup.sh — TA-Lib C-library fallback builder
# ─────────────────────────────────────────────────────────────────────────────
# YOU PROBABLY DO NOT NEED THIS SCRIPT.
#
# As of the current requirements.txt, TA-Lib is pinned to >=0.6.7, which ships
# PREBUILT manylinux cp311/cp312 wheels on PyPI. On a standard Colab runtime
# (Python 3.11/3.12, Linux x86_64) `pip install TA-Lib` therefore downloads a
# binary wheel and needs NO C library and NO compilation. The notebook's Cell 3
# (`apt-get install -y -q ta-lib; pip install -r requirements.txt`) works as-is.
#
# THIS SCRIPT IS THE FALLBACK for the rare platform where no prebuilt TA-Lib
# wheel exists for the running Python/arch (e.g. a brand-new Python release, a
# non-x86_64 arch, or a locked-down image). In that case the TA-Lib Python
# wrapper must compile against the TA-Lib *C* library, which Ubuntu/Colab does
# NOT provide via apt. This script downloads and builds that C library from
# source, installs it, runs ldconfig, then pip-installs the wrapper.
#
# USAGE (only if `pip install TA-Lib` fails to find a wheel):
#     bash scripts/colab_setup.sh
#     # then re-run:  pip install -r requirements.txt
#
# Idempotent: safe to re-run. Requires sudo/root (Colab runs as root).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

TALIB_VERSION="0.4.0"                 # canonical TA-Lib C source release
SRC_URL="https://downloads.sourceforge.net/project/ta-lib/ta-lib/${TALIB_VERSION}/ta-lib-${TALIB_VERSION}-src.tar.gz"
BUILD_DIR="$(mktemp -d)"

echo "[colab_setup] Installing build toolchain (build-essential, wget)…"
apt-get update -y -q
apt-get install -y -q build-essential wget

echo "[colab_setup] Downloading TA-Lib C source ${TALIB_VERSION}…"
cd "${BUILD_DIR}"
wget -q "${SRC_URL}" -O ta-lib-src.tar.gz
tar -xzf ta-lib-src.tar.gz
cd "ta-lib"

# The 0.4.0 config.guess predates x86_64 detection on some images; refresh it so
# ./configure recognizes the host triplet. Non-fatal if the download fails.
echo "[colab_setup] Refreshing config.guess/config.sub (host detection)…"
wget -q -O config.guess "https://git.savannah.gnu.org/cgit/config.git/plain/config.guess" || true
wget -q -O config.sub   "https://git.savannah.gnu.org/cgit/config.git/plain/config.sub"   || true

echo "[colab_setup] Configuring + building TA-Lib C library (this takes ~1 min)…"
./configure --prefix=/usr
make -j"$(nproc)"
make install

echo "[colab_setup] Running ldconfig so the dynamic linker finds libta_lib…"
ldconfig

echo "[colab_setup] Installing the TA-Lib Python wrapper against the built C lib…"
# Point the build at the headers/libs we just installed under /usr.
export TA_INCLUDE_PATH="/usr/include"
export TA_LIBRARY_PATH="/usr/lib"
pip install --no-binary :all: "TA-Lib>=0.6.7,<0.7"

echo "[colab_setup] Verifying import…"
python -c "import talib; print('[colab_setup] TA-Lib OK, version', talib.__version__)"

echo "[colab_setup] Done. You can now run: pip install -r requirements.txt"
