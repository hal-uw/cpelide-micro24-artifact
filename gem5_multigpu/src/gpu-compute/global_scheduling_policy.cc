#include "gpu-compute/global_scheduling_policy.hh"

#include <map>

#include "debug/GlobalScheduler.hh"
#include "gpu-compute/global_scheduler.hh"

Event*
RRGlobalPolicy::chooseKernel(GlobalScheduler *gs, uint32_t qID)
{
    if (gs->qInfo[qID]->hasKernels())
    {
        readIdx = gs->qInfo[qID]->oldestKernIdx();
        writeIdx = readIdx+1;
        bool isCopy = false;
        
        if(gs->qInfo[qID]->kernels[readIdx].kernKey.kernelName.length() <= 1){
            DPRINTF(GlobalScheduler, "For this new kernel readIdx is %d and number of kernels left are %d\n", readIdx, gs->qInfo[qID]->kernels.size());
            gs->qInfo[qID]->kernels.erase(readIdx);
            return nullptr;
        }
        
        for( int i= 0 ; i < gs->qInfo[qID]->kernels[readIdx].num_gpus; i++){
        gpu_id = nextGPU;
        isCopy = (i >0) ? true : false;
        nextGPU = ((gpu_id - STARTING_GPU_ID + 1) % gs->num_gpus)
                  + STARTING_GPU_ID;
        DPRINTF(GlobalScheduler, "Adding gpu_id %d to list of GPUs with num_gpus = %d and isCopy is %d\n", gpu_id, gs->qInfo[qID]->kernels[readIdx].num_gpus, isCopy);
        gs->markKernsForDisp(qID, readIdx, writeIdx, gpu_id, isCopy);
        
        }
        return new GlobalScheduler::GSSendKernelEvent(gs, gpu_id, qID,
                                                      writeIdx, readIdx,
                                                      readIdx, false);
    } else {
        return nullptr;
    }
}

Event*
RRCSGlobalPolicy::chooseKernel(GlobalScheduler *gs, uint32_t qID)
{
    if (gs->qInfo[qID]->hasKernels())
    {
        readIdx = gs->qInfo[qID]->oldestKernIdx();
        writeIdx = readIdx+1;
        bool isCopy = false;
        
        if(gs->qInfo[qID]->kernels[readIdx].kernKey.kernelName.length() <= 1){
            DPRINTF(GlobalScheduler, "For this new kernel readIdx is %d and number of kernels left are %d\n", readIdx, gs->qInfo[qID]->kernels.size());
            gs->qInfo[qID]->kernels.erase(readIdx);
            return nullptr;
        }
        
        for( int i= 0 ; i < gs->qInfo[qID]->kernels[readIdx].num_gpus; i++){
        
        isCopy = (i >0) ? true : false;
        gpu_id = (!isCopy) ? STARTING_GPU_ID : nextGPU;
        nextGPU = ((gpu_id - STARTING_GPU_ID + 1) % gs->num_gpus)
                  + STARTING_GPU_ID;
        DPRINTF(GlobalScheduler, "Adding gpu_id %d to list of GPUs with num_gpus = %d and isCopy is %d\n", gpu_id, gs->qInfo[qID]->kernels[readIdx].num_gpus, isCopy);
        gs->markKernsForDisp(qID, readIdx, writeIdx, gpu_id, isCopy);
        
        }
        return new GlobalScheduler::GSSendKernelEvent(gs, gpu_id, qID,
                                                      writeIdx, readIdx,
                                                      readIdx, false);
    } else {
        return nullptr;
    }
}

Event*
NormalGlobalPolicy::chooseKernel(GlobalScheduler *gs, uint32_t qID)
{
    if (gs->qInfo[qID]->hasKernels())
    {
        readIdx = gs->qInfo[qID]->oldestKernIdx();
        writeIdx = gs->qInfo[qID]->newestKernIdx() + 1;

        gpu_id = nextGPU;
        DPRINTF(GlobalScheduler, "In normal global policy");
        gs->markKernsForDisp(qID, readIdx, writeIdx, gpu_id, false);
        return new GlobalScheduler::GSSendKernelEvent(gs, gpu_id, qID,
                                                      writeIdx, readIdx,
                                                      readIdx, false);

    } else {
        return nullptr;
    }
}

Event*
LaxityGlobalPolicy::chooseKernel(GlobalScheduler *gs, uint32_t qID)
{
    std::map<uint32_t, Tick> laxMap;
    for (auto &it : gs->qInfo)
    {
        auto &queue = it.second;
        // Want to look at passed in queue no matter what
        // as we're scheduling before the kernel in said queue finishes
        if (queue->hasKernels() && (queue->numDispatched() == 0 ||
            (it.first == qID && queue->numDispatched() <= 1))) {
            if (queue->numDispatched() == 1 &&
                queue->dispKernels.begin()->second.dispStatus != D) {
                continue;
            }

            laxMap[it.first] = 0;

            // DurationTime part of algo
            if (queue->jobStarted()) {
                laxMap[it.first] = curTick() - queue->getJobStart();
            }

            // TimeRemaining part of algo
            for (const auto& kern : queue->getKernelIdxs())
            {
                laxMap[it.first] += gs->estimateKernelRuntime(it.first, kern,
                                                              gpu_id);
            }
            DPRINTF(GlobalScheduler, "Queue %d: laxity: %d\n", it.first,
                    laxMap[it.first]);
        }
    }

    // Get max as most pressing queue is one with highest
    // DurationTime + TimeRemaining when we don't have Deadline
    auto maxIt = *std::max_element(laxMap.begin(), laxMap.end(),
                    [&] (const auto& a, const auto& b) {
                        if (a.second == b.second)
                        {
                            return gs->qInfo[a.first]->getPriority() <
                                   gs->qInfo[b.first]->getPriority();
                        }
                        return a.second < b.second;
                    });

    // Based on the numDispatched() == 0 condition from above, we know
    // that we can dispatch the queue with worst deadline
    if (laxMap.find(maxIt.first) != laxMap.end()) {
        uint32_t queueIdx = maxIt.first;
        readIdx = gs->qInfo[queueIdx]->oldestKernIdx();
        writeIdx = readIdx + 1;

        DPRINTF(GlobalScheduler, "Selecting kernel %d from Q[%d]\n", readIdx,
                queueIdx);

        // Simplifying g-score as numAvailableWFs for now
        std::vector<int32_t> gpuWFs;
        gpuWFs = gs->availableWFs;

        // If we're dispatching from the same queue we passed in (qID)
        // We need to update the GPU WF estimate to assume the currently
        // executing kernel from said queue is finished
        if (gs->qInfo[qID]->numDispatched() != 0) {
            // Ideally we'd only ever see 2 in flight at once:
            // One being run, one about to be run. We'd also
            // only ever see them on the same GPU as well.
            uint32_t dispKern = gs->qInfo[qID]->dispatchedKernIdx();
            uint32_t dispGpu = gs->qInfo[qID]->getKernelGPU(dispKern) -
                               STARTING_GPU_ID;

            DPRINTF(GlobalScheduler, "GPU[%d] Updating gpuWF estimate by %d "
                    "WFs (is %d currently)\n", dispGpu,
                    gs->kernelWFs(qID, dispKern), gpuWFs[dispGpu]);
            gpuWFs[dispGpu] += gs->kernelWFs(qID, dispKern);
        }

        auto mostWFs = std::max_element(gpuWFs.begin(),
                                        gpuWFs.end());

        DPRINTF(GlobalScheduler, "GPU[%d] has most WFs (%d) available\n",
                std::distance(gpuWFs.begin(), mostWFs), *mostWFs);

        // If we don't have enough WFs to dispatch a single WG,
        // don't dispatch
        if (*mostWFs < (int32_t)gs->kernelWGSize(queueIdx, readIdx)) {
            gs->recordSchedulingEvent(EVENTS::FAIL_GPU);
            return nullptr;
        }

        gpu_id = std::distance(gpuWFs.begin(), mostWFs) +
                 STARTING_GPU_ID;

        gs->markKernsForDisp(queueIdx, readIdx, writeIdx, gpu_id, false);
        return new GlobalScheduler::GSSendKernelEvent(gs, gpu_id, queueIdx,
                                                      writeIdx, readIdx,
                                                      readIdx, false);
    }

    gs->recordSchedulingEvent(EVENTS::FAIL_KERN);
    return nullptr;
}
