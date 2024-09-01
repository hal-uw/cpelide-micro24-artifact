#!/bin/bash

# Step-by-step guide to setting up Docker environment for CPElide MICRO '24
# artifact.
# All code assumes script is run from the root directory the artifact was cloned
# in

# Step 0: set path
export GEM5_PATH=${PWD} # benchmarks need this path to be set
# some systems require Docker to be initialized first
systemctl --user start docker.service
# clone benchmarks subrepo in properly
git submodule update --init

# Step 1: setup docker image -- this will take 30-45 minutes
cd gem5_multigpu/multigpu_benchmarks
docker build -t cpelide-artifact .

# Step 2: build gem5
cd ../ # goes back to gem5_multgpu folder
# compiles gem5 with as many threads as there are cores on machine -- this will
# take 10-60 minutes depending on your machine
# Optional: if you want gem5 to not print anything to screen when compiling, use this version
#docker run --rm -v ${PWD}:${PWD} -w ${PWD} cpelide-artifact scons -sQ -j$(nproc) build/GCN3_X86/gem5.opt
docker run --rm -v ${PWD}:${PWD} -w ${PWD} cpelide-artifact scons -j$(nproc) build/GCN3_X86/gem5.opt
# build m5ops that benchmarks use for annotations
cd util/m5
docker run --rm -v ${PWD}:${PWD} -w ${PWD} cpelide-artifact scons build/x86/out/m5
cd ../..

# Step 3: build benchmarks (datasets already grabbed from clone)
# Note: some benchmarks are commented out to have this complete in a reasonable
# amount of time -- uncomment them if you want the full set

# babelstream
docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/babelstream cpelide-artifact bash -c 'make'

# DeepBench (RNNs)
# docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/DeepBench/code/amd cpelide-artifact bash -c 'make'

# gemm
# docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/gemm cpelide-artifact bash -c 'make'

# HACC
# Note: HACC requires setting numerous environment variables to run correctly.  To
# avoid needing to set all of these, we instead build a docker for it, which
# has all these variables pre-set in its Dockerfile
# docker build -t hacc-test multigpu_benchmarks/halo-finder
# docker run -rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/halo-finder/src hacc-test bash -c 'make hip/ForceTreeTest'

# LULESH
# docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/lulesh cpelide-artifact bash -c 'make'

# Pannotia -- compile all in a single command
# docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/pannotia cpelide-artifact bash -c 'bash ./buildall.sh gem5-fusion'

# Pennant
# docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/pennant cpelide-artifact bash -c 'make'

# Rodinia (BFS, Backprop, BTree, Gaussian, Hotspot3D, Hotspot, LUD, DWT2D, NW, Pathfinder, SRAD_v2)
#for bench in bfs, backprop, b+tree, gaussian, hotspot3D, hotspot, lud, dwt2d, nw, pathfinder
#do
#    docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/rodinia/$bench cpelide-artifact bash -c 'make'
#done

# SRAD v2 has an extra level of indirection
# docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/rodinia/srad/srad_v2 cpelide-artifact bash -c 'make'

# Square
docker run --rm -v $(pwd):$(pwd) -w $(pwd)/multigpu_benchmarks/square cpelide-artifact bash -c 'make square_m'
