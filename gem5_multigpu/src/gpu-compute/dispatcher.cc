/*
 * Copyright (c) 2011-2015,2018 Advanced Micro Devices, Inc.
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


#include "gpu-compute/dispatcher.hh"

#include "debug/GPUAgentDisp.hh"
#include "debug/GPUDisp.hh"
#include "debug/GPUKernelInfo.hh"
#include "debug/GPUWgLatency.hh"
#include "debug/GlobalScheduler.hh"
#include "dev/hsa/hw_scheduler.hh"
#include "gpu-compute/global_scheduler.hh"
#include "gpu-compute/gpu_command_processor.hh"
#include "gpu-compute/hsa_queue_entry.hh"
#include "gpu-compute/shader.hh"
#include "gpu-compute/wavefront.hh"
#include "sim/syscall_emul_buf.hh"
#include "sim/system.hh"
#define STARTING_GPU_ID 2765
GPUDispatcher::GPUDispatcher(const Params &p)
    : SimObject(p), shader(nullptr), gpuCmdProc(nullptr),
      tickEvent([this]{ exec(); },
          "GPU Dispatcher tick", false, Event::CPU_Tick_Pri),
      dispatchActive(false), gpu_id(p.gpu_id), gsThreshold(p.threshold), stats(this)
{
    schedule(&tickEvent, 0);
}

GPUDispatcher::~GPUDispatcher()
{
}

HSAQueueEntry*
GPUDispatcher::hsaTask(int disp_id)
{
    assert(hsaQueueEntries.find(disp_id) != hsaQueueEntries.end());
    return hsaQueueEntries[disp_id];
}

void
GPUDispatcher::setCommandProcessor(GPUCommandProcessor *gpu_cmd_proc)
{
    gpuCmdProc = gpu_cmd_proc;
}

void
GPUDispatcher::setShader(Shader *new_shader)
{
    shader = new_shader;
}

void
GPUDispatcher::serialize(CheckpointOut &cp) const
{
    Tick event_tick = 0;

    if (tickEvent.scheduled())
        event_tick = tickEvent.when();

    SERIALIZE_SCALAR(event_tick);
}

void
GPUDispatcher::unserialize(CheckpointIn &cp)
{
    Tick event_tick;

    if (tickEvent.scheduled())
        deschedule(&tickEvent);

    UNSERIALIZE_SCALAR(event_tick);

    if (event_tick) {
        schedule(&tickEvent, event_tick);
    }
}

/**
 * After all relevant HSA data structures have been traversed/extracted
 * from memory by the CP, dispatch() is called on the dispatcher. This will
 * schedule a dispatch event that, when triggered, will attempt to dispatch
 * the WGs associated with the given task to the CUs.
 */
void
GPUDispatcher::dispatch(HSAQueueEntry *task)
{
    ++stats.numKernelLaunched;

    DPRINTF(GPUDisp, "launching kernel: %s, dispatch ID: %d\n",
            task->kernelName(), task->dispatchId());
    DPRINTF(GPUAgentDisp, "launching kernel: %s, dispatch ID: %d\n",
            task->kernelName(), task->dispatchId());

    //execIds.push(task->dispatchId());
    dispatchActive = true;
    hsaQueueEntries.emplace(task->dispatchId(), task);

    taskIds.push(TaskStruct(task->dispatchId(), task->getPriority(), 0));

    if (!tickEvent.scheduled()) {
        schedule(&tickEvent, curTick() + shader->clockPeriod());
    }
}

void
GPUDispatcher::exec()
{
    int fail_count(0);
    int disp_count(0);

    /**
     * There are potentially multiple outstanding kernel launches.
     * It is possible that the workgroups in a different kernel
     * can fit on the GPU even if another kernel's workgroups cannot
     */
    DPRINTF(GPUDisp, "Launching %d Kernels\n", taskIds.size());
    DPRINTF(GPUAgentDisp, "Launching %d Kernels\n", taskIds.size());

    if (taskIds.size() > 0) {
        ++stats.cyclesWaitingForDispatch;
    }

    /**
     * dispatch work cannot start until the kernel's invalidate is
     * completely finished; hence, kernel will always initiates
     * invalidate first and keeps waiting until inv done
     */
    //while (execIds.size() > fail_count) {
    while (taskIds.size() > fail_count) {
        //int exec_id = execIds.front();

        TaskStruct next = taskIds.top();
        uint32_t exec_id = next.ID;
        uint32_t priority = next.priority;
        uint32_t order = next.order;

        auto task = hsaQueueEntries[exec_id];
        bool launched(false);

        // acq is needed before starting dispatch
        if (shader->impl_kern_launch_acq) {
            // try to invalidate cache
            shader->prepareInvalidate(task, global_scheduler->getInvalidateFlushControl(shader->gpu_id - STARTING_GPU_ID,  task->globalKernId(), task->globalQId(), true) );
        }    
         else {
            // kern launch acquire is not set, skip invalidate
            task->markInvDone();
        }
        
        if(global_scheduler->getInvalidateFlushControl(shader->gpu_id - STARTING_GPU_ID,  task->globalKernId(), task->globalQId(), false) && !shader->impl_kern_end_rel){
            shader->prepareFlush(task);
        }

        else if (!global_scheduler->getInvalidateFlushControl(shader->gpu_id - STARTING_GPU_ID,  task->globalKernId(), task->globalQId(), false) && !shader->impl_kern_end_rel) {
            task->markWbDone();
        }
        /**
         * invalidate is still ongoing, put the kernel on the queue to
         * retry later
         */
        if (!task->isInvDone() || (!(shader->impl_kern_end_rel) &&  !task->isWbDone()) || !global_scheduler->isFlushL2Done( task->globalKernId(), task->globalQId())){
            //execIds.push(exec_id);
            //order++;
            taskIds.push(TaskStruct(exec_id, priority, order));

            ++fail_count;

            DPRINTF(GPUDisp, "kernel %d failed to launch, due to [%d] pending"
                " invalidate requests and [%d] flush requests\n", exec_id, task->outstandingInvs(), task->outstandingWbs());

            // try the next kernel_id
            //execIds.pop();
            taskIds.pop();
             
            continue;
        }

        // kernel invalidate is done, start workgroup dispatch
        while (!task->dispComplete()) {
            // update the thread context
            shader->updateContext(task->contextId());

            // attempt to dispatch workgroup
            DPRINTF(GPUWgLatency, "Attempt Kernel Launch cycle:%d kernel:%d\n",
                curTick(), exec_id);

            if (!shader->dispatchWorkgroups(task)) {
                /**
                 * if we failed try the next kernel,
                 * it may have smaller workgroups.
                 * put it on the queue to retry later
                 */
                DPRINTF(GPUDisp, "kernel %d failed to launch\n", exec_id);
                DPRINTF(GPUKernelInfo, "kernel %d failed to launch\n",
                                       exec_id);
                //execIds.push(exec_id);
                //order++;
                taskIds.push(TaskStruct(exec_id, priority, order));

                ++fail_count;
                break;
            } else if (!launched) {
                launched = true;
                disp_count++;
                DPRINTF(GPUKernelInfo, "Launched kernel %d\n", exec_id);
            }
        }

        // try the next kernel_id
        //execIds.pop();
        taskIds.pop();
    }

    DPRINTF(GPUDisp, "Returning %d Kernels\n", doneIds.size());
    DPRINTF(GPUWgLatency, "Kernel Wgs dispatched: %d | %d failures\n",
            disp_count, fail_count);

    while (doneIds.size()) {
        DPRINTF(GPUDisp, "Kernel %d completed\n", doneIds.front());
        doneIds.pop();
    }
}

bool
GPUDispatcher::isReachingKernelEnd(Wavefront *wf)
{
    int kern_id = wf->kernId;
    assert(hsaQueueEntries.find(kern_id) != hsaQueueEntries.end());
    auto task = hsaQueueEntries[kern_id];
    assert(task->dispatchId() == kern_id);

    /**
     * whether the next workgroup is the final one in the kernel,
     * +1 as we check first before taking action
     */
    return (task->numWgCompleted() + 1 == task->numWgChipletTotal());
}

/**
 * update the counter of oustanding inv requests for the kernel
 * kern_id: kernel id
 * val: +1/-1, increment or decrement the counter (default: -1)
 */
void
GPUDispatcher::updateInvCounter(int kern_id, int val) {
    assert(val == -1 || val == 1);

    auto task = hsaQueueEntries[kern_id];
    task->updateOutstandingInvs(val);

    // kernel invalidate is done, schedule dispatch work
    if (task->isInvDone() && !tickEvent.scheduled()) {
        schedule(&tickEvent, curTick() + shader->clockPeriod());
    }
}

/**
 * update the counter of oustanding wb requests for the kernel
 * kern_id: kernel id
 * val: +1/-1, increment or decrement the counter (default: -1)
 *
 * return true if all wbs are done for the kernel
 */
bool
GPUDispatcher::updateWbCounter(int kern_id, int val) {
    assert(val == -1 || val == 1);

    auto task = hsaQueueEntries[kern_id];
    task->updateOutstandingWbs(val);

    // true: WB is done, false: WB is still ongoing
    return (task->outstandingWbs() == 0);
}

/**
 * get kernel's outstanding cache writeback requests
 */
int
GPUDispatcher::getOutstandingWbs(int kernId) {
    auto task = hsaQueueEntries[kernId];

    return task->outstandingWbs();
}

/**
 * When an end program instruction detects that the last WF in
 * a WG has completed it will call this method on the dispatcher.
 * If we detect that this is the last WG for the given task, then
 * we ring the completion signal, which is used by the CPU to
 * synchronize with the GPU. The HSAPP is also notified that the
 * task has completed so it can be removed from its task queues.
 */
void
GPUDispatcher::notifyWgCompl(Wavefront *wf)
{
    int kern_id = wf->kernId;
    
    auto task = hsaQueueEntries[kern_id];
   
    assert(task->dispatchId() == kern_id);
    task->notifyWgCompleted();
     DPRINTF(GPUDisp, "notify WgCompl %d completed WGs are %d\n", wf->wgId, task->numWgCompleted() );
    DPRINTF(GPUWgLatency, "WG Complete cycle:%d wg:%d kernel:%d cu:%d\n",
        curTick(), wf->wgId, kern_id, wf->computeUnit->cu_id);

    global_scheduler->kernelWgFinish(task->globalQId(),
                                     task->globalKernId(),
                                     wf->wgId);

    if (task->numWgCompleted() == task->numWgChipletTotal() ) {
        // Notify the HSA PP that this kernel is complete
        gpuCmdProc->hsaPacketProc()
            .finishPkt(task->dispPktPtr(), task->queueId());
        if (task->completionSignal() && task->getChipletId() == 1) {
            /**
            * HACK: The semantics of the HSA signal is to decrement
            * the current signal value. We cheat here and read out
            * he value from main memory using functional access and
            * then just DMA the decremented value.
            */
            uint64_t signal_value =
                gpuCmdProc->functionalReadHsaSignal(task->completionSignal());

            DPRINTF(GPUDisp, "HSA AQL Kernel Complete with completion "
                    "signal! Addr: %d the signal value is %d and numWGCompleted is %d and chiplet ID is %d\n", task->completionSignal(), signal_value, task->numWgCompleted() , task->getChipletId());

            gpuCmdProc->updateHsaSignal(task->completionSignal(),
                                        signal_value - 1);
        } else {
            DPRINTF(GPUDisp, "HSA AQL Kernel Complete! No completion "
                "signal\n");
        }

        DPRINTF(GPUWgLatency, "Kernel Complete ticks:%d kernel:%d\n",
                curTick(), kern_id);
        DPRINTF(GPUKernelInfo, "Completed kernel %d\n", kern_id);

        DPRINTF(GlobalScheduler, "Queue[%d] Kernel[%d] complete.\n",
                task->globalQId(), task->globalKernId());
        global_scheduler->kernelComplete(task->globalQId(),
                                         task->globalKernId());
    } else if (task->numWgCompleted() == int(task->numWgChipletTotal()*gsThreshold)) {
        // Notify scheduler we will need more work soon
        // May need to model delay? Not sure
        DPRINTF(GlobalScheduler, "Queue[%d] Kernel[%d] almost done "
                "(%d/%d WGs), requesting more work.\n", task->globalQId(),
                task->globalKernId(), task->numWgCompleted(),
                task->numWgChipletTotal());
        global_scheduler->recordSchedulingEvent(EVENTS::GPU_REQ,
            task->globalQId(), 0, task->globalKernId());

        global_scheduler->markKernDispatched(task->globalQId(),
                                             task->globalKernId());

        global_scheduler->makeSchedulingDecision(task->globalQId(),
                                                 false, true);
    }

    if (!tickEvent.scheduled()) {
        schedule(&tickEvent, curTick() + shader->clockPeriod());
    }
}

void
GPUDispatcher::scheduleDispatch()
{
    if (!tickEvent.scheduled()) {
        schedule(&tickEvent, curTick() + shader->clockPeriod());
    }
}

GPUDispatcher::GPUDispatcherStats::GPUDispatcherStats(Stats::Group *parent)
    : Stats::Group(parent),
      ADD_STAT(numKernelLaunched, "number of kernel launched"),
      ADD_STAT(cyclesWaitingForDispatch, "number of cycles with outstanding "
               "wavefronts that are waiting to be dispatched")
{
}

void
GPUDispatcher::attachGlobalScheduler(GlobalScheduler* glb_scheduler){
    global_scheduler = glb_scheduler;
}

void GPUDispatcher::notifyMemSyncCompletion(int queue_id, int kernel_id, int chiplet_id , bool inv_or_wb){
      global_scheduler->notifyMemSyncCompletion(queue_id, kernel_id, chiplet_id ,inv_or_wb);
    }