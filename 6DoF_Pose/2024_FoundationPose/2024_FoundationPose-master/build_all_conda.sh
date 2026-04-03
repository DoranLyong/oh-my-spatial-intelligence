PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

#-- Install mycpp
#cd ${PROJ_ROOT}/mycpp/ && \
#rm -rf build && mkdir -p build && cd build && \
#cmake .. && \
#make -j$(nproc)
cd ${PROJ_ROOT}/mycpp/ && \
rm -rf build *.so *.egg-info dist && \
uv pip install --no-build-isolation -e .

#-- Install mycuda
cd ${PROJ_ROOT}/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
#python -m pip install -e .
uv pip install --no-build-isolation -e .

cd ${PROJ_ROOT}
