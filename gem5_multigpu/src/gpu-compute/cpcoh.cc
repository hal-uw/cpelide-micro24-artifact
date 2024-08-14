// File: cpcoh.cpp
// Author: Rajesh Shashi Kumar (HAL Research Group)
// Email: rshashikumar@wisc.edu

#include "base/trace.hh"
#include "debug/CPCoh.hh"
#include <unordered_map>
#include <list>
#include <utility>
#include <iostream>
#include "gpu-compute/cpcoh.hh"

using namespace std;

CpCoh::CpCoh(uint32_t capacity)
{
	m_capacity = capacity;
	cpcohReset();
	invalidate_queue.reset();
	flush_queue.reset();
}

void CpCoh::cpcohReset()
{
	dsid_map.clear(); // Drop all entries from the hash map
	for (int i = 0; i < NUM_CHIPLET; i++)
		chiplet_cache[i].clear(); // Drop all entries from the chiplet_cache vectors
}

/* Retrieve horizontal chiplet vector table entry */
chipletVector CpCoh::getcpcohEntry(uint64_t dsID)
{
	chipletVector current;
	if (dsid_map.count(dsID) == 0)
	{
		std::fill_n(current.begin(), current.size(), CPCOH_NOT_PRESENT);
		return current; // 0 indicates not found and assume that it is not stored in the table in this state
	}
	uint32_t idx = dsid_map[dsID]; // Index into the hash map and return the chiplet vector
	chipletVector lookup;
	for (int i = 0; i < NUM_CHIPLET; i++)
		lookup[i] = chiplet_cache[i].at(idx);
	return lookup;
}

/*
Assume two things are received from the scheduler:
1. Valid vector
2. Mode of access
*/
std::pair<chipletID, chipletID> CpCoh::putcpcohEntry(schedulerVector const &sv)
{
	invalidate_queue.reset();
	flush_queue.reset();
	for (auto current_sv : sv) // Iterate through each of the arguments eg. A, B, C
	{
		uint64_t dsID = get<0>(current_sv);
		chipletVector schedule = get<1>(current_sv);
		chipletVector mode = get<2>(current_sv);
		bool identical_reuse = get<3>(current_sv);

		DPRINTF(CPCoh, "Received from Scheduler> DSID:%lx,Schedule0:%d,Mode0:%d\n", dsID, (int)(schedule[0].to_ulong()), (int)(mode[0].to_ulong()));
		DPRINTF(CPCoh, "Received from Scheduler> DSID:%lx,Schedule1:%d,Mode1:%d\n", dsID, (int)(schedule[1].to_ulong()), (int)(mode[1].to_ulong()));
		DPRINTF(CPCoh, "Received from Scheduler> DSID:%lx,Schedule2:%d,Mode2:%d\n", dsID, (int)(schedule[2].to_ulong()), (int)(mode[2].to_ulong()));
		DPRINTF(CPCoh, "Received from Scheduler> DSID:%lx,Schedule3:%d,Mode3:%d\n", dsID, (int)(schedule[3].to_ulong()), (int)(mode[3].to_ulong()));

		chipletVector new_cv; // Create new chiplet vector that is blank

		for (uint32_t i = 0; i < new_cv.size(); i++)
		{
			new_cv[i] = schedule[i] ^ mode[i]; // Fill in the chiplet vector with the information from mode/schedule
											   // DEBUG: std::cout << "Chiplet" << i << "Schedule:" << schedule[i] << "Mode:" << mode[i] << "New CV:" << new_cv[i] << std::endl;
		}

		if (dsid_map.count(dsID) == 0) // If entry does not exist already
		{
			for (int i = 0; i < NUM_CHIPLET; i++)
				chiplet_cache[i].push_back(new_cv[i]);	  // Do this for all NUM_CHIPLET (4) vectors
			dsid_map[dsID] = chiplet_cache[0].size() - 1; // Index of the last inserted element
		}
		else
		{
			if (identical_reuse){ // Skip maintenance operations on existing data structuresif identical reuse is present across kernels
				//  If dsID exists, update values for those entries in the chiplet cache vectors
				cpcohMaintainReuse(dsID, new_cv); // Perform maintenance on old values preserving reuse
			} 
			else {
				cpcohMaintain(dsID, new_cv); // Perform maintenance on old values
			}
		}
	}

	// Coalesce flush and invalidation based on queue. Invalidations must preceed flush otherwise reuse is lost
	for (std::size_t i = 0; i < invalidate_queue.size(); ++i)
		if (invalidate_queue[i])
		{
			std::cout << "Invalidate chiplet num:" << i << std::endl; // Send invalidation for this chiplet
			DPRINTF(CPCoh, "Invalidate chiplet num: %d\n", i);
		}

	for (std::size_t i = 0; i < flush_queue.size(); ++i)
		if (flush_queue[i])
		{

			std::cout << "Flush chiplet:" << i << std::endl; // cacheFlush(current.dirty); Placeholder for later implementation
			DPRINTF(CPCoh, "Flush chiplet num: %d\n", i);
		}

	chipletID invalidate_queue_per_kernel = invalidate_queue;
	chipletID flush_queue_per_kernel = flush_queue;

	invalidate_queue.reset();
	flush_queue.reset();
	printcpcohTable();
	return std::make_pair(invalidate_queue_per_kernel, flush_queue_per_kernel);
}

void CpCoh::cpcohMaintain(uint64_t dsID, chipletVector new_cv)
{
	// Get the information from table on current state of data structure
	chipletVector old_cv = getcpcohEntry(dsID);
	// Get the index in the dsID vector, which will be used to index to entries in the other vertical cache vectors
	uint64_t dsid_idx = dsid_map[dsID];

	// Number of chiplets where the data is currently dirty for this data structure
	uint32_t count_dirty = cpcohcountDirty(old_cv);

	// Flush and stale decisions need updated state for affected caches based on what is sent to Command Processor
	bool flush_condition_met = false;
	bool stale_condition_met = false;

	// Horizontally (across chiplets) traverse each entry of the new chipletVector and compare it with
	// the existing entry to find out if flushes or invalidations are necessary
	for (std::uint32_t i = 0; i < new_cv.size(); i += 1)
	{
		/* Check if INVALIDATION is necessary based on new kernel to be scheduled */

		// Scheduled on a particular chiplet && the specific chiplet holds the structure in stale state
		if ((new_cv[i] == CPCOH_VALID || new_cv[i] == CPCOH_DIRTY) && (old_cv[i] == CPCOH_STALE))
		{
			invalidate_queue.set(i); // Send invalidation for this chiplet
			// Replicate the consequence of above action for my local reference as well
			old_cv[i] = CPCOH_NOT_PRESENT;
			// Perform cache-wide (vertically) operation on invalidation to update expected new states
			for (std::uint32_t idx = 0; idx < chiplet_cache[i].size(); idx++)
			{
				if (chiplet_cache[i].at(idx) == CPCOH_STALE || chiplet_cache[0].at(idx) == CPCOH_VALID)
				{
					chiplet_cache[i].at(idx) = CPCOH_NOT_PRESENT;
				}
			}
		}

		/* Check if FLUSH is necessary based on new kernel to be scheduled */

		// Case 1: If a single chiplet held this DS in dirty state previously, and if our new schedule matches, then we avoid flush promoting REUSE
		if (count_dirty == 1)
			if ((new_cv[i] == CPCOH_VALID || new_cv[i] == CPCOH_DIRTY) && (old_cv[i] != CPCOH_DIRTY))
			{
				flush_condition_met = true;
			}

		// Case 2: More than one chiplet has this DS in dirty state previously
		// If dirty on more than one, we need to FLUSH anyway because we DO NOT track which subset is dirty on which chiplet
		// TODO: This can be further optimized once we have range-based invalidations in place
		if (count_dirty > 1)
			flush_condition_met = true;
	}

	for (std::uint32_t i = 0; i < new_cv.size(); i += 1)
	{
		// Flush operation across caches
		if (flush_condition_met)
			if (old_cv[i] == CPCOH_DIRTY) // Flush every chiplet that has it in the dirty state
			{
				flush_queue.set(i); // Send flush for this chiplet
				// Replicate the consequence of above action for my local reference as well
				old_cv[i] = CPCOH_VALID;
				// Vertically update the cache states on FLUSH
				for (std::uint32_t idx = 0; idx < chiplet_cache[i].size(); idx++)
				{
					if (chiplet_cache[i].at(idx) == CPCOH_DIRTY)
					{
						chiplet_cache[i].at(idx) = CPCOH_VALID;
					}
				}
			}

		/* Update actual table based on the predicted updates that the new kernel will perform */
		if (new_cv[i] == CPCOH_DIRTY)
		{
			chiplet_cache[i].at(dsid_idx) = CPCOH_DIRTY;
			// When the new kernel is going to leave a chiplet in the dirty state, the updated chipletVector
			// should mark other chiplets currently in VALID as STALE because we cannot have VALID & DIRTY simultaneously
			stale_condition_met = true;
		}

		if (new_cv[i] == CPCOH_VALID)
		{
			chiplet_cache[i].at(dsid_idx) = CPCOH_VALID;
		}
	}

	/* Marking stale across caches based on new_cv */
	for (std::uint32_t i = 0; i < new_cv.size(); i += 1)
	{
		if (stale_condition_met)
			// New kernel RW, and now I check chiplets that previously had it in VALID rendered stale because the new chiplet did not map to the same one
			if (old_cv[i] == CPCOH_VALID && !(new_cv[i] == CPCOH_VALID || new_cv[i] == CPCOH_DIRTY))
			{
				chiplet_cache[i].at(dsid_idx) = CPCOH_STALE;
			}
	}
}

void CpCoh::cpcohMaintainReuse(uint64_t dsID, chipletVector new_cv)
{
	// CAVEAT! Assume the schedule is the same as the previous kernel for now. Feb 28, 2023

	// Get the information from table on current state of data structure
	// chipletVector old_cv = getcpcohEntry(dsID);
	// Get the index in the dsID vector, which will be used to index to entries in the other vertical cache vectors
	uint64_t dsid_idx = dsid_map[dsID];

	for (std::uint32_t i = 0; i < new_cv.size(); i += 1)
	{
		/* Update actual table based on the predicted updates that the new kernel will perform */
		// Valid -> Dirty transition
		if (new_cv[i] == CPCOH_DIRTY)
		{
			chiplet_cache[i].at(dsid_idx) = CPCOH_DIRTY;
		}
	}
}

uint32_t CpCoh::cpcohcountDirty(chipletVector &cv)
{
	std::uint32_t count = 0;

	for (std::uint32_t i = 0; i < cv.size(); i += 1)
	{
		if (cv[i] == CPCOH_DIRTY)
		{ // Dirty state
			count += 1;
		}
	}
	return count;
}

// Function to display contents of cache
void CpCoh::printcpcohTable()
{
	// Iterate in the map and print all the elements in each of the chiplet caches
	for (unordered_map<uint32_t, uint32_t>::iterator iter = dsid_map.begin(); iter != dsid_map.end(); ++iter)
	{
		uint32_t k = iter->first;
		uint32_t v = iter->second;
		// std::cout << "DSID:" << k << ","; // Print DSID
		DPRINTF(CPCoh, "DSID:%llx\n", k);
		for (uint32_t i = 0; i < NUM_CHIPLET; i++)
		{
			// std::cout << chiplet_cache[i].at(v) << ",";
			DPRINTF(CPCoh, "Chiplet%d:%d\n", i, (int)((chiplet_cache[i].at(v)).to_ulong()));
		}
		// std::cout << std::endl;
	}
}
