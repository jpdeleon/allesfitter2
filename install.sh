#!/bin/bash
"""
allesfitter requires ellc which contains fortran code wrapped in python.
If `pip install -r requirements.txt && pip install -e .` didn't work,
then try ``
"""
echo "=== Installing packages from requirements.txt ==="
pip install -r requirements.txt

echo "=== Uninstalling ellc ==="
pip uninstall ellc -y

echo "=== Cloning the ellc repository ==="
git clone https://github.com/pmaxted/ellc.git

echo "=== Building the ellc package (Fortran extensions) ==="
cd ellc
python setup.py build_ext --inplace

echo "=== Installing ellc package into the Python environment ==="
pip install -e .

cd ..

echo "=== Installing allesfitter ==="
pip install -e .

echo "=== allesfitter installation complete ==="
echo "=== To test, run the script `prepare_allesfit` ==="

