#!/bin/bash
# This script collects cache and runtime stats from rundir. It takes the max of shader active ticks for multiGPU setup.
# Usage: (1) Ensure all the result directories are in a file named "file_list" (2) Ensure path to cpcohStatsCalc.py script is correct (3) ./collect_stats.sh

if [ ! -e "file_list" ]; then
  find . -maxdepth 1 -type d ! -path . -printf '%P\n' | sed '/^\.$/d' > file_list
fi

while read j; do
	touch summary_$j
        echo $j
	printf "$j\n" >> summary_$j
	python3 ./cpcohStatsCalc.py $j/stats.txt 4 >> summary_$j
	printf "\n Top 3 kernel ticks:\n" >> summary_$j
	grep "system.cpu80.shaderActiveTicks" $j/stats.txt | awk '{print $2}' | head -n 3 > cpu80ticks_$j
	grep "system.cpu81.shaderActiveTicks" $j/stats.txt | awk '{print $2}' | head -n 3 > cpu81ticks_$j
	grep "system.cpu82.shaderActiveTicks" $j/stats.txt | awk '{print $2}' | head -n 3 > cpu82ticks_$j
	grep "system.cpu83.shaderActiveTicks" $j/stats.txt | awk '{print $2}' | head -n 3 > cpu83ticks_$j
	paste cpu80ticks_$j cpu81ticks_$j cpu82ticks_$j cpu83ticks_$j| awk '{if ($1 > $2) print $1; else print $2}' >> summary_$j
	rm cpu80ticks_$j cpu81ticks_$j cpu82ticks_$j cpu83ticks_$j
done < file_list
