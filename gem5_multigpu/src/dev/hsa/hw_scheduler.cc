/*
 * Copyright (c) 2016-2017 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * For use for simulation and test purposes only
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "dev/hsa/hw_scheduler.hh"

#include "debug/HSAPacketProcessor.hh"
#include "gpu-compute/global_scheduler.hh"
#include "mem/packet_access.hh"

#define HWSCHDLR_EVENT_DESCRIPTION_GENERATOR(XEVENT) \
  const char*                                    \
  HWScheduler::XEVENT::description() const       \
  {                                              \
      return #XEVENT;                            \
  }

HWSCHDLR_EVENT_DESCRIPTION_GENERATOR(SchedulerWakeupEvent)

void
HWScheduler::SchedulerWakeupEvent::process()
{
    hwSchdlr->wakeup();
}

void
HWScheduler::wakeup()
{
    // The scheduler unmaps an idle queue from the
    // registered qList and maps a new queue
    // to the registered list from the active list.
    // For this implementation, an idle queue means
    // a queue that does not have any outstanding dispatch
    // at the time of this scheduler's wakeup

    bool wakeup = contextSwitchQ();
    schedWakeup(wakeup);
}

void
HWScheduler::schedWakeup(bool wakeup)
{
    // If atleast there is one queue that is not registered
    // then wakeup again
    if (!schedWakeupEvent.scheduled() && wakeup) {
        hsaPP->schedule(&schedWakeupEvent, curTick() + wakeupDelay);
        DPRINTF(HSAPacketProcessor,
                "Scheduling wakeup at %lu\n", (curTick() + wakeupDelay));
    }
}

void
HWScheduler::registerNewQueue(uint64_t hostReadIndexPointer,
                              uint64_t basePointer,
                              uint64_t queue_id,
                              uint32_t size)
{
    assert(queue_id < MAX_ACTIVE_QUEUES);
    // Map queue ID to doorbell.
    // We are only using offset to pio base address as doorbell
    // We use the same mapping function used by hsa runtime to do this mapping
    //
    // Originally
    // #define VOID_PTR_ADD32(ptr,n)
    //     (void*)((uint32_t*)(ptr) + n)/*ptr + offset*/
    // (Addr)VOID_PTR_ADD32(0, queue_id)
    Addr db_offset = sizeof(uint32_t)*queue_id;
    if (dbMap.find(db_offset) != dbMap.end()) {
        panic("Creating an already existing queue (queueID %d)", queue_id);
    }

    // Populate doorbell map
    dbMap[db_offset] = queue_id;

    if (queue_id >= MAX_ACTIVE_QUEUES) {
        panic("Attempting to create a queue (queueID %d)" \
              " beyond PIO range", queue_id);
    }

    HSAQueueDescriptor* q_desc =
       new HSAQueueDescriptor(basePointer, db_offset,
                              hostReadIndexPointer, size);
    AQLRingBuffer* aql_buf =
        new AQLRingBuffer(NUM_DMA_BUFS, hsaPP->name());
    QCntxt q_cntxt(q_desc, aql_buf);
    activeList[dbMap[db_offset]] = q_cntxt;

    priority_to_al_id.insert(std::pair<uint32_t, uint32_t> (7, queue_id));

    // Check if this newly created queue can be directly mapped
    // to registered queue list
    //M5_VAR_USED bool register_q = mapQIfSlotAvlbl(queue_id, aql_buf, q_desc);
    //schedWakeup();
    DPRINTF(HSAPacketProcessor,
             "%s: offset = %p, qID = %d, AL size %d\n",
             __FUNCTION__, db_offset, queue_id,
             dbMap.size());
}

bool
HWScheduler::findEmptyHWQ()
{
    DPRINTF(HSAPacketProcessor,
            "Trying to find empty HW queue, @ %s\n", __FUNCTION__);
    if (regdListMap.size() < hsaPP->numHWQueues) {
        for (int emptyQId = 0; emptyQId < hsaPP->numHWQueues; emptyQId++) {
            HSAQueueDescriptor* qDesc =
                hsaPP->getRegdListEntry(nextRLId)->qCntxt.qDesc;
            // If qDesc is empty, we find an empty HW queue
            if (qDesc == NULL) {
                return true;
            }
            nextRLId = (nextRLId + 1) % hsaPP->numHWQueues;
        }
        // We should be able to find an empty slot in registered list
        // So, we should not reach here
        panic("Cannot find empty queue\n");
    }
    return false;
}

bool
HWScheduler::mapQIfSlotAvlbl(uint32_t q_id, AQLRingBuffer* aql_buf,
                             HSAQueueDescriptor* q_desc)
{
    DPRINTF(HSAPacketProcessor,
            "Trying to map new queue, @ %s\n", __FUNCTION__);
    if (!findEmptyHWQ()) {
        return false;
    }
    addQCntxt(q_id, aql_buf, q_desc);
    scheduleAndWakeupMappedQ();
    updateRRVars(q_id, nextRLId);
    return true;
}

void
HWScheduler::scheduleAndWakeupMappedQ()
{
    // There maybe AQL packets in the mapped queue waiting
    // to be fetched. Invoke the logic to fetch AQL packets
    hsaPP->getCommandsFromHost(0, nextRLId);
    // Schedule the newly mapped queue
    if (hsaPP->regdQList[nextRLId]->dispPending())
        hsaPP->schedAQLProcessing(nextRLId);
}

void
HWScheduler::addQCntxt(uint32_t al_idx, AQLRingBuffer* aql_buf,
                        HSAQueueDescriptor* q_desc)
{
    assert(hsaPP->getRegdListEntry(nextRLId)->qCntxt.qDesc == NULL);
    assert(hsaPP->getRegdListEntry(nextRLId)->qCntxt.aqlBuf == NULL);
    // Move the context
    hsaPP->getRegdListEntry(nextRLId)->qCntxt.qDesc = q_desc;
    hsaPP->getRegdListEntry(nextRLId)->qCntxt.aqlBuf = aql_buf;
    // Add the mapping to registered list map
    regdListMap[al_idx] = nextRLId;
    reg_to_queue_id[nextRLId] = al_idx;
    priority_to_reg_id.insert(std::pair<uint32_t, uint32_t>
                              (aql_buf->priority(), nextRLId));
    DPRINTF(HSAPacketProcessor, "Mapped HSA queue %d to hw queue %d: @ %s\n",
            al_idx, nextRLId, __FUNCTION__);
}

bool
HWScheduler::contextSwitchQ()
{
    DPRINTF(HSAPacketProcessor,
            "Trying to map next queue, @ %s\n", __FUNCTION__);
    // Identify the next queue, if there is nothing to
    // map, return false
    if (!findNextActiveALQ()) {
        return false;
    }
    // If there is empty slot available, use that slot
    if (mapQBasedOnPriority(nextALId)) {
        return true;
    }
    // There is no empty slot to map this queue. So, we need to
    // unmap a queue from registered list and find a slot.
    // If nothing can be unmapped now, return false
    // One queue is unmapped from registered list and that queueID
    // is stored in nextRLId. We will map this queue to that unmapped slot
    return false;
}

void
HWScheduler::updateRRVars(uint32_t al_idx, uint32_t rl_idx)
{
    nextALId = (al_idx + 1) % MAX_ACTIVE_QUEUES;
    nextRLId = (rl_idx + 1) % hsaPP->numHWQueues;
}

bool
HWScheduler::unmapQFromRQ()
{
    // Identify the next idle queue, if there is no
    // idle queue, we cannot unmap
    if (!findNextIdleRLQ()) {
        return false;
    }
    removeQCntxt();
    return true;
}

void
HWScheduler::removeQCntxt()
{
    // The nextRLId gives the registered queue that is to be unmapped.
    // We can find the corresponding queue_id from the doorbellPointer
    Addr db_offset =
        hsaPP->getRegdListEntry(nextRLId)->qCntxt.qDesc->doorbellPointer;
    hsaPP->getRegdListEntry(nextRLId)->qCntxt.qDesc = NULL;
    hsaPP->getRegdListEntry(nextRLId)->qCntxt.aqlBuf = NULL;
    // Here, we are unmappping a queue wihtout waiting for the outstanding
    // dependency signal reads to complete. We will discard any outstanding
    // reads and will reset the signal values here.
    hsaPP->getRegdListEntry(nextRLId)->depSignalRdState.discardRead = true;
    hsaPP->getRegdListEntry(nextRLId)->depSignalRdState.resetSigVals();
    uint32_t al_idx = dbMap[db_offset];
    assert(regdListMap[al_idx] == nextRLId);
    // Unmap from regdListMap.
    regdListMap.erase(al_idx);

    assert(reg_to_queue_id[nextRLId]==al_idx);
    reg_to_queue_id.erase(nextRLId);

    eraseMultiMapRL(nextRLId);
}

bool
HWScheduler::findNextActiveALQ()
{
    for (int activeQId = 0; activeQId < MAX_ACTIVE_QUEUES; activeQId++) {
        uint32_t al_id = (nextALId + activeQId) % MAX_ACTIVE_QUEUES;
        auto aqlmap_iter = activeList.find(al_id);
        if (aqlmap_iter != activeList.end()) {
            // If this queue is already mapped
            if (regdListMap.find(al_id) != regdListMap.end()) {
                continue;
            } else {
                DPRINTF(HSAPacketProcessor,
                        "Next Active ALQ %d (current %d), max ALQ %d\n",
                         al_id, nextALId, MAX_ACTIVE_QUEUES);
                nextALId = al_id;
                return true;
            }
        }
    }
    return false;
}

bool
HWScheduler::findNextIdleRLQ()
{
    for (int regdQId = 0; regdQId < hsaPP->numHWQueues; regdQId++) {
        uint32_t rl_idx = (nextRLId + regdQId) % hsaPP->numHWQueues;
        if (isRLQIdle(rl_idx)) {
            nextRLId = rl_idx;
            return true;
        }
    }
    return false;
}

// This function could be moved to packet processor
bool
HWScheduler::isRLQIdle(uint32_t rl_idx)
{
    DPRINTF(HSAPacketProcessor,
            "@ %s, analyzing hw queue %d\n", __FUNCTION__, rl_idx);
    HSAQueueDescriptor* qDesc = hsaPP->getRegdListEntry(rl_idx)->qCntxt.qDesc;

    // If there a pending DMA to this registered queue
    // then the queue is not idle
    if (qDesc->dmaInProgress) {
        return false;
    }

    // Since packet completion stage happens only after kernel completion
    // we need to keep the queue mapped till all the outstanding kernels
    // from that queue are finished
    if (hsaPP->inFlightPkts(rl_idx)) {
        return false;
    }

    return true;
}

void
HWScheduler::write(Addr db_addr, uint32_t doorbell_reg)
{
    auto dbmap_iter = dbMap.find(db_addr);
    if (dbmap_iter == dbMap.end()) {
        panic("Writing to a non-existing queue (db_offset %x)", db_addr);
    }
    uint32_t al_idx = dbMap[db_addr];
    // Modify the write pointer
    activeList[al_idx].qDesc->writeIndex = doorbell_reg;
     // If a queue is unmapped and remapped (common in full system) the qDesc
    // gets reused. Keep the readIndex up to date so that when the HSA packet
    // processor gets commands from host, the correct entry is read after
    // remapping
    activeList[al_idx].qDesc->readIndex = doorbell_reg - 1;
    if (regdListMap.find(al_idx) != regdListMap.end()) {
        hsaPP->getCommandsFromHost(0, regdListMap[al_idx]);
    } else {
        if (!searchHighestPriority()) {
            DPRINTF(HSAPacketProcessor, "no queue\n");
            panic("no queue\n");
        }
        mapQBasedOnPriority(nextALId);
    }
    schedWakeup(regdListMap.size() < activeList.size());
}

void
HWScheduler::unregisterQueue(uint64_t queue_id)
{
    // Pointer arithmetic on a null pointer is undefined behavior. Clang
    // compilers therefore complain if the following reads:
    // `(Addr)(VOID_PRT_ADD32(0, queue_id))`
    //
    // Originally
    // #define VOID_PTR_ADD32(ptr,n)
    //     (void*)((uint32_t*)(ptr) + n)/*ptr + offset*/
    // (Addr)VOID_PTR_ADD32(0, queue_id)
    Addr db_offset = sizeof(uint32_t)*queue_id;
    auto dbmap_iter = dbMap.find(db_offset);
    if (dbmap_iter == dbMap.end()) {
        panic("Destroying a non-existing queue (db_offset %x)",
               db_offset);
    }
    uint32_t al_idx = dbMap[db_offset];
    assert(dbMap[db_offset] == dbmap_iter->second);
    if (!activeList[al_idx].qDesc->isEmpty()) {
        // According to HSA runtime specification says, deleting
        // a queue before it is fully processed can lead to undefined
        // behavior and it is the application's responsibility to
        // avoid this situation.
        // Even completion signal is not a sufficient indication for a
        // fully processed queue; for example completion signal may be
        // asserted when a read pointer update is in progress
        warn("Destroying a non-empty queue");
    }
    delete activeList[al_idx].qDesc;
    delete activeList[al_idx].aqlBuf;
    activeList.erase(al_idx);
    // Unmap doorbell from doorbell map
    dbMap.erase(db_offset);
    eraseMultiMapAL(al_idx);

    if (regdListMap.find(al_idx) != regdListMap.end()) {
        uint32_t rl_idx = regdListMap[al_idx];
        hsaPP->getRegdListEntry(rl_idx)->qCntxt.aqlBuf = NULL;
        hsaPP->getRegdListEntry(rl_idx)->qCntxt.qDesc = NULL;
        hsaPP->getRegdListEntry(rl_idx)->depSignalRdState.discardRead = true;
        hsaPP->getRegdListEntry(rl_idx)->depSignalRdState.resetSigVals();
        assert(!hsaPP->getRegdListEntry(rl_idx)->aqlProcessEvent.scheduled());
        regdListMap.erase(al_idx);
        reg_to_queue_id.erase(rl_idx);
        eraseMultiMapRL(rl_idx);
        // A registered queue is released, let us try to map
        // a queue to that slot
        contextSwitchQ();
    }
    schedWakeup(regdListMap.size() < activeList.size());
}

void
HWScheduler::updateQueuePriority(uint32_t queue_id, uint32_t queue_priority)
{
    eraseMultiMapAL(queue_id);

    activeList[queue_id].aqlBuf->setPriority(queue_priority);
    priority_to_al_id.insert(std::pair<uint32_t, uint32_t>
        (activeList[queue_id].aqlBuf->priority(), queue_id));

    //check if already mapped
    auto it = regdListMap.find(queue_id);
    if (it != regdListMap.end()) {
        removeQCntxtFromId(regdListMap[queue_id]);
    }

    if (!searchHighestPriority()) {
        DPRINTF(HSAPacketProcessor, "no queue\n");
        //panic("no queue\n");
        return;
    }

    mapQBasedOnPriority(nextALId);

}

bool
HWScheduler::searchHighestPriority()
{
    std::multimap<uint32_t, uint32_t>::reverse_iterator it;
    for (it = priority_to_al_id.rbegin(); it != priority_to_al_id.rend(); ++it)
    {
        uint32_t al_idx = it->second;
        // If this queue is already mapped
        if (regdListMap.find(al_idx) != regdListMap.end()) {
            continue;
        }
        if (!isDoorbellWritten(al_idx) && !isALQReady(al_idx)) {
            continue;
        }
        nextALId = al_idx;
        return true;
    }
    return false;
}

uint32_t
HWScheduler::getLowestPriority()
{
    std::multimap<uint32_t, uint32_t>::iterator it;
    uint32_t rl_idx = -1;
    for (it = priority_to_reg_id.begin(); it != priority_to_reg_id.end(); ++it)
    {
        uint32_t rl_idx = it->second;
        if (isRLQIdle(rl_idx) && !isScheduled(rl_idx)) {
            return rl_idx;
        }
    }
    return rl_idx;
}

uint32_t
HWScheduler::getHighestPriority()
{
    std::multimap<uint32_t, uint32_t>::reverse_iterator it;
    uint32_t rl_idx = -1;
    for (it = priority_to_reg_id.rbegin(); it != priority_to_reg_id.rend();
         ++it)
    {
        uint32_t rl_idx = it->second;
        if (isRLQIdle(rl_idx)) {
            if (isRLQReady(rl_idx) &&
                !(hsaPP->regdQList[rl_idx]->isDmaInProgress()))
            {
                return rl_idx;
            }
        }
    }
    return rl_idx;
}

uint32_t
HWScheduler::isDoorbellWritten(uint32_t al_idx)
{
    return (activeList[al_idx].qDesc->writeIndex
            - activeList[al_idx].qDesc->readIndex);
}

bool
HWScheduler::isRLQReady(uint32_t rl_idx)
{
    DPRINTF(HSAPacketProcessor,
            "@ %s, analyzing hw queue %d\n", __FUNCTION__, rl_idx);

    if (hsaPP->readyDispPkts(rl_idx)) {
        return true;
    }

    return false;
}

bool
HWScheduler::isScheduled(uint32_t rl_idx)
{
    return hsaPP->getRegdListEntry(rl_idx)->isScheduled();
}

bool
HWScheduler::isALQReady(uint32_t al_idx)
{

    if (activeList[al_idx].aqlBuf->dispPending()) {
        assert(activeList[al_idx].aqlBuf->dispIdx()
               == activeList[al_idx].aqlBuf->rdIdx());
        return true;
    }

    return false;
}

bool
HWScheduler::mapQBasedOnPriority(uint32_t queue_id) {
    uint32_t priority = activeList[queue_id].aqlBuf->priority();

    if (!findEmptyHWQ()) {
        //check lowest priority di RL
        uint32_t rl_idx = getLowestPriority();
        if ((rl_idx != -1)
            && (hsaPP->regdQList[rl_idx]->getPriority() < priority))
        {
            nextRLId = rl_idx;
            removeQCntxt();
        }
        else {
            return false;
        }
    }

    // One queue is unmapped from registered list and that queueID
    // is stored in nextRLId. We will map this queue to that unmapped slot
    addQCntxt(queue_id, activeList[queue_id].aqlBuf,
              activeList[queue_id].qDesc);
    scheduleAndWakeupMappedQ();
    updateRRVars(queue_id, nextRLId);
    return true;
}

bool
HWScheduler::eraseMultiMapAL(uint32_t al_idx)
{
    std::multimap<uint32_t, uint32_t>::iterator it;
    for (it = priority_to_al_id.begin(); it != priority_to_al_id.end(); it++)
    {
        if (it->second == al_idx) {
            priority_to_al_id.erase(it);
            return true;
        }
    }
    return false;
}

bool
HWScheduler::eraseMultiMapRL(uint32_t rl_idx)
{
    std::multimap<uint32_t, uint32_t>::iterator it;
    for (it = priority_to_reg_id.begin(); it != priority_to_reg_id.end(); it++)
    {
        if (it->second == rl_idx) {
            priority_to_reg_id.erase(it);
            return true;
        }
    }
    return false;
}

void
HWScheduler::removeQCntxtFromId(uint32_t rl_idx)
{
    DPRINTF(HSAPacketProcessor, "Remove Queue %d from RL\n", rl_idx);
    hsaPP->getRegdListEntry(rl_idx)->qCntxt.qDesc = NULL;
    hsaPP->getRegdListEntry(rl_idx)->qCntxt.aqlBuf = NULL;
    // Here, we are unmappping a queue wihtout waiting for the outstanding
    // dependency signal reads to complete. We will discard any outstanding
    // reads and will reset the signal values here.
    hsaPP->getRegdListEntry(rl_idx)->depSignalRdState.discardRead = true;
    hsaPP->getRegdListEntry(rl_idx)->depSignalRdState.resetSigVals();

    uint32_t al_idx = reg_to_queue_id[rl_idx];
    assert(regdListMap[al_idx] == rl_idx);
    // Unmap from regdListMap.
    regdListMap.erase(al_idx);

    assert(reg_to_queue_id[rl_idx]==al_idx);
    reg_to_queue_id.erase(rl_idx);

    eraseMultiMapRL(rl_idx);
}

void
HWScheduler::write_packet(uint32_t queue_id, uint32_t writeIndex,
                          uint32_t readIndex, uint32_t dispIndex)
{
    uint32_t al_idx = queue_id;
    // Modify the write pointer
    assert(writeIndex > 0);
    activeList[al_idx].qDesc->writeIndex = writeIndex;
    activeList[al_idx].qDesc->readIndex = readIndex;
    activeList[al_idx].aqlBuf->setWrIdx(readIndex);
    activeList[al_idx].aqlBuf->setRdIdx(readIndex);
    activeList[al_idx].aqlBuf->setDispIdx(dispIndex);
    // If this queue is mapped, then start DMA to fetch the
    // AQL packet
    if (regdListMap.find(al_idx) != regdListMap.end()) {
        hsaPP->getCommandsFromHost(0, regdListMap[al_idx]);
    } else{
        if (!searchHighestPriority()) {
            DPRINTF(HSAPacketProcessor, "no queue\n");
            panic("no queue\n");
        }
        mapQBasedOnPriority(nextALId);
    }
    schedWakeup(regdListMap.size() < activeList.size());
}

void
HWScheduler::attachGlobalScheduler(GlobalScheduler* glb_scheduler)
{
     global_scheduler = glb_scheduler;
}
