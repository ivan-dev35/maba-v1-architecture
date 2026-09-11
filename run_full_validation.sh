#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/6] Auditing model parameters..."
python3 tests/verify_params.py

echo "[2/6] Running unit tests..."
python3 tests/test_components.py
python3 tests/test_scaling.py

echo "[3/6] Testing MTP speculative generation..."
python3 tests/test_speculative_generation.py

echo "[4/6] Testing training loop..."
python3 tests/test_e2e_training.py

echo "[5/6] Building C++ engine..."
mkdir -p cpp/build
cd cpp/build
cmake ..
make -j$(nproc)
cd "$SCRIPT_DIR"

echo "[6/6] Validating numerical equivalence (C++ vs PyTorch)..."
python3 generate_reference.py
cpp/build/test_numerical maba_ref.bin ref_logits.bin

echo "Running C++ CLI benchmark..."
cpp/build/maba_cli maba_ref.bin

echo "Validation successful."
rm -f maba_ref.bin ref_logits.bin
