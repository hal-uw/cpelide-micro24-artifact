// File: CpCoh.h
// Author: Rajesh Shashi Kumar (HAL Research Group)
// Email: rshashikumar@wisc.edu

#ifndef CPCOH_H
#define CPCOH_H

#include <iostream>
#include <bitset>
#include <unordered_map>
#include <vector>
#include <utility>
#include <array>
#include <tuple>

using namespace std;

// Sizes
#define NUM_GPU 1
#define NUM_CHIPLET_PER_GPU 4
#define NUM_CHIPLET NUM_CHIPLET_PER_GPU *NUM_GPU
#define NUM_TABLE_ENTRIES 10

// States
#define CPCOH_NOT_PRESENT 0b00
#define CPCOH_VALID 0b01
#define CPCOH_DIRTY 0b10
#define CPCOH_STALE 0b11

// Types
typedef std::bitset<2> bitVector;                                              // 2bits per chiplet
typedef std::array<bitVector, NUM_CHIPLET> chipletVector;                      // 2 * NUM_CHIPLET
typedef vector<tuple<uint32_t, chipletVector, chipletVector, bool>> schedulerVector; // dsID, schedule, mode
typedef std::bitset<NUM_CHIPLET> chipletID;

// Class that defines the CPCoh table
class CpCoh
{
private:
    uint32_t m_capacity;                                           // Maximum number of entries in the CpCoh
    std::array<std::vector<bitVector>, NUM_CHIPLET> chiplet_cache; // Vertically: Caches represented as vectors and an array of them (4)
    std::unordered_map<uint32_t, uint32_t> dsid_map;               // Horizontally: Maps Data Structure ID to Vector Index for chiplet Vector lookups

    chipletID flush_queue;
    chipletID invalidate_queue;

public:
    CpCoh(uint32_t capacity); // Constructor

    /* CpCoh management */
    void cpcohReset();                          // Clear all entries on cache reset signal
    chipletVector getcpcohEntry(uint64_t dsID); // Look up sharing status of a data structure for performing cache ops
    std::pair<chipletID, chipletID> putcpcohEntry( schedulerVector const &sv);    // Insert new entry upon new kernel, Evictions when CpCoh is full, Update when entry already exists
    // Automatically called when inserting a new entry into the table
    void cpcohMaintain(uint64_t dsID, chipletVector schedule); // Invokes flush/invalidate on eviction/schedule
    void cpcohMaintainReuse(uint64_t dsID, chipletVector schedule);
    /* Cache operations */
    void cacheInvalidate(chipletVector c); // Invalidate at cache granularity of L2 on specific chiplet
    void cacheFlush(chipletVector c);      // Invalidate at cache granularity

    /* Helper functions */
    uint32_t cpcohcountDirty(chipletVector &c); 

    /* Debug */
    void printcpcohTable(); // Function to display contents of cache
};

#endif

/*
Notes:

1. For now, we limit ourselves to a single GPU. TBD: Tracking across multi-GPU
2. Invalidate at data structure granularity will need separate cache operation functions
3. How do we have implicit trigger of maintain when entry is missing in cache due to L2-eviction
*/
