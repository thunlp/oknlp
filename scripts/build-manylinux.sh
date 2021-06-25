#!/bin/bash
set -e -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat manylinux_2_12_x86_64 -w dist/
    fi
}


# Compile wheels
for PYBIN in /opt/python/cp3*/bin; do
    "${PYBIN}/pip" wheel . --no-deps -w build_dist
done

# Bundle external shared libraries into the wheels
for whl in build_dist/*.whl; do
    repair_wheel "$whl"
done
