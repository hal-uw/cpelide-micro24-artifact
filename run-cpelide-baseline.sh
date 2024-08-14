#!/bin/bash
# Running experiments in Docker environment for CPElide MICRO '24 artifact.
# Note: some benchmarks are commented out to have this complete in a reasonable
# amount of time -- uncomment them if you want the full set
# This script specifically runs the baseline configuration from the methodology

# Square Baseline
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_square configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=4096kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS --benchmark-root=multigpu_benchmarks/square -c square_m --options="524288 1 2 2048 256" --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# Babelstream Baseline
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_babelstream configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=4096kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS --benchmark-root=multigpu_benchmarks/babelstream -c BabelStream_lk --options="524288" --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2 --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# DeepBench (RNNs)
# Left out because of long runtime

# gemm
# Left out because of long runtime

# HACC Baseline
# Note: HACC requires setting numerous environment variables to run correctly.  To
# avoid needing to set all of these, we instead build a docker for it, which
# has all these variables pre-set in its Dockerfile
# We assume run-cpelide.sh is run after setup-cpelide.sh and thus the docker
# already exists.
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --outdir=baseline_hacc configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/halo-finder/src/hip/  -c ForceTreeTest --options="0.5 0.1 512 0.1 2 N 12 rcb"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel |& tee baseline_hacc.log 

# LULESH - BASELINE 
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt  --debug-flags=GPUKernelInfo --debug-file=run_cpcoh.log --outdir=baseline_lulesh configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS --benchmark-root=multigpu_benchmarks/lulesh/bin --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2 -c lulesh --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# PENNANT - Baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt  --debug-flags=GPUKernelInfo --debug-file=run_cpcoh.log --outdir=baseline_pennant configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS --benchmark-root=multigpu_benchmarks/pennant/build/ --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2 -c pennant --options="multigpu_benchmarks/pennant/test/noh/noh.pnt" --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# SSSP - Baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --outdir=baseline_sssp configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/pannotia/sssp/bin/  -c sssp.gem5 --options="multigpu_benchmarks/pannotia/sssp/NH_75.gr 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# Color large baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --outdir=baseline_large_color configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS --benchmark-root=multigpu_benchmarks/pannotia/color/bin/ --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2 -c color_maxmin.gem5 --options="multigpu_benchmarks/pannotia/color/AK_25.gr 0" --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# FW baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --outdir=cpcoh_fw configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/pannotia/fw/bin/  -c fw_hip.gem5 --options="multigpu_benchmarks/pannotia/fw/512_65536.gr"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB

# BTREE
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_btree configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/b+tree/  -c b+tree.out --options="file multigpu_benchmarks/rodinia/b+tree/mil.txt command multigpu_benchmarks/rodinia/b+tree/command.txt 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel |& tee btree_baseline_large.log  &

# BFS - baseline
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_bfs configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/bfs/  -c bin/bfs --options="multigpu_benchmarks/rodinia/bfs/graph65536.txt 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# BACKPROP - baseline
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_backprop configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/backprop/  -c bin/backprop --options="65536 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# DWT2D - baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_dwt2d configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/dwt2d/  -c dwt2d --options="multigpu_benchmarks/rodinia/dwt2d/192.bmp -d 192x192 -f -5 -l 3 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# GAUSSIAN
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_gaussian configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/gaussian/  -c gaussian --options="-f multigpu_benchmarks/rodinia/gaussian/matrix208.txt 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# HOTSPOT
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_hotspot configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/hotspot/  -c hotspot --options="512 2 20 multigpu_benchmarks/rodinia/hotspot/temp_512 multigpu_benchmarks/rodinia/hotspot/power_512 multigpu_benchmarks/rodinia/hotspot/output.out 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# HOTSPOT3D
docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_hotspot3D configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/hotspot3D/  -c 3D --options="512 8 100 multigpu_benchmarks/rodinia/hotspot3D/power_512x8 multigpu_benchmarks/rodinia/hotspot/temp_512x8 multigpu_benchmarks/rodinia/hotspot/output.out"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# LUD - baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_lud configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/lud/hip/  -c lud_hip --options="-i multigpu_benchmarks/rodinia/lud/512.dat 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# NW - baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_nw configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/nw/  -c needle --options="2048 10"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# PATHFINDER - baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_pathfinder configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/pathfinder/  -c pathfinder --options="100000 100 20 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel

# SRAD v2 - baseline
# docker run --rm --volume $(pwd):$(pwd) -w $(pwd) cpelide-artifact build/GCN3_X86/gem5.opt --debug-flags=GlobalScheduler,CPCoh --debug-file=run_cpcoh.log --outdir=results_baseline_large_srad_v2 configs/example/apu_se.py -n80 -u60 --cu-per-sa=60 --num-gpu-complex=1 --reg-alloc-policy=dynamic --barriers-per-cu=16 --num-tccs=8 --bw-scalor=8 --tcc-size=8192kB --tcc-assoc=32 --num-dirs=64 --mem-size=16GB --mem-type=HBM_1000_4H_1x64 --vreg-file-size=16384 --sreg-file-size=800 --num-hw-queues=256 --num-gpus=4 --gs-policy=GSP_RRCS  --benchmark-root=multigpu_benchmarks/rodinia/srad/srad_v2/  -c srad --options="2048 2048 0 127 0 127 0.5 2 1 0"  --coal-tokens=160 --gpu-clock=1801MHz --ruby-clock=1000MHz --TCC_latency=121 --vrf_lm_bus_latency=6 --mem-req-latency=69 --mem-resp-latency=69 --TCP_latency=16 --gs-num-sched-gpu=2  --max-coalesces-per-cycle=10 --sqc-size=16kB --default-acq-rel
