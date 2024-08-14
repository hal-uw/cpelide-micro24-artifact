#!/bin/bash

# Running experiments in Docker environment for CPElide MICRO '24 artifact.
# Note: some benchmarks are commented out to have this complete in a reasonable
# amount of time -- uncomment them if you want the full set

bash ./run-cpelide-cpcoh.sh
bash ./run-cpelide-baseline.sh

