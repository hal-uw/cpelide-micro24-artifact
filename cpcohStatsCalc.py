#Author: <Redacted for anonymity>
#Usage python script <Relative File Path> <num_sched_gpu>

import os
import subprocess
from sys import argv
accesses = 0
hits = 0
misses = 0
currentMaxTicks = 0
Ticks = [] 
for i in range(0, int(argv[2])):
    Ticks.append(subprocess.check_output(f"grep system.cpu8{i}.shaderActiveTicks {argv[1]} | tr -s ' ' | cut -f2 -d ' ' ", shell=True).split())
    tickChiplet = 0
    for elem in Ticks[-1]:
        tickChiplet += int(elem)
    currentMaxTicks = max(tickChiplet, currentMaxTicks)    

for i in range(0, int(argv[2])*8):
    accesses += int(subprocess.check_output(f"grep -ri 'system.ruby.tcc_cntrl{i}.L2cache.m_demand_accesses' {argv[1]} | tail -1 | tr -s ' ' |  cut -f2 -d ' ' ", shell=True))
    hits += int(subprocess.check_output(f"grep -ri 'system.ruby.tcc_cntrl{i}.L2cache.m_demand_hits' {argv[1]} | tail -1 | tr -s ' ' |  cut -f2 -d ' ' ", shell=True))
    misses += int(subprocess.check_output(f"grep -ri 'system.ruby.tcc_cntrl{i}.L2cache.m_demand_misses' {argv[1]} | tail -1 | tr -s ' ' |  cut -f2 -d ' ' ", shell=True))
print(f"Accesses = {accesses}\n")
print(f"Hits = {hits}\n")
print(f"Misses = {misses}\n")
print(f"Hit rate is {(hits/accesses)*100}%")
print(f"Total Runtime is {currentMaxTicks}")
