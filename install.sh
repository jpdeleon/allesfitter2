#!/bin/bash

if command -v uv &> /dev/null; then
    PIP="uv pip install"
    PIP_EDIT="uv pip install -e ."
else
    PIP="pip install"
    PIP_EDIT="pip install -e ."
fi

echo "=== Installing allesfitter dependencies ==="
$PIP .

echo "=== Uninstalling ellc ==="
$PIP uninstall ellc -y

ELLCPATH="../ellc"

echo "=== Cloning the ellc repository into $ELLCPATH ==="
git clone https://github.com/pmaxted/ellc.git "$ELLCPATH"

echo "=== Building the ellc package (Fortran extensions) ==="
cd "$ELLCPATH"
python setup.py build_ext --inplace

echo "=== Installing ellc package into the Python environment ==="
$PIP_EDIT .

cd -

echo "=== Installing allesfitter ==="
$PIP_EDIT .

echo "=== allesfitter installation complete ==="
echo "=== To test, run the script \`prepare_allesfit\` ==="
