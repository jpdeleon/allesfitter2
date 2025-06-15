#!/bin/bash

echo "=== Installing packages from requirements.txt ==="
pip install -r requirements.txt

echo "=== Uninstalling ellc ==="
pip uninstall ellc -y

# Clone into the parent directory
ELLCPATH="../ellc"

echo "=== Cloning the ellc repository into $ELLCPATH ==="
git clone https://github.com/pmaxted/ellc.git "$ELLCPATH"

echo "=== Building the ellc package (Fortran extensions) ==="
cd "$ELLCPATH"
python setup.py build_ext --inplace

echo "=== Installing ellc package into the Python environment ==="
pip install -e .

# Return to the original directory
cd -

echo "=== Installing allesfitter ==="
pip install -e .

echo "=== allesfitter installation complete ==="
echo "=== To test, run the script \`prepare_allesfit\` ==="
