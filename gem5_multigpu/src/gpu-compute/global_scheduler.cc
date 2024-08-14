#include "gpu-compute/global_scheduler.hh"

#include <algorithm>
#include <functional>

#include "base/chunk_generator.hh"
#include "base/compiler.hh"
#include "debug/CPCoh.hh"
#include "debug/GlobalScheduler.hh"
#include "debug/HSAPacketProcessor.hh"
#include "dev/hsa/hsa_packet.hh"
#include "dev/hsa/hsa_signal.hh"
#include "dev/hsa/hw_scheduler.hh"
#include "gpu-compute/cpcoh.hh"
#include "gpu-compute/dispatcher.hh"
#include "gpu-compute/gpu_command_processor.hh"
#include "gpu-compute/gpu_static_inst.hh"
#include "gpu-compute/shader.hh"
#include "gpu-compute/wavefront.hh"
#include "mem/packet_access.hh"
#include "sim/process.hh"
#include "sim/proxy_ptr.hh"
#include "sim/syscall_emul_buf.hh"
#include "sim/system.hh"

#define PKT_TYPE(PKT) \
    ((hsa_packet_type_t)(((PKT->header) >> HSA_PACKET_HEADER_TYPE) \
        & (HSA_PACKET_HEADER_WIDTH_TYPE - 1)))

#define GS_EVENT_DESCRIPTION_GENERATOR(XEVENT) \
  const char*                                    \
  GlobalScheduler::XEVENT::description() const       \
  {                                              \
      return #XEVENT;                            \
  }

GS_EVENT_DESCRIPTION_GENERATOR(GSDmaEvent)
GS_EVENT_DESCRIPTION_GENERATOR(GSSendKernelEvent)
GS_EVENT_DESCRIPTION_GENERATOR(GSDecideKernelEvent)
GS_EVENT_DESCRIPTION_GENERATOR(GSReceiveEvent)

GlobalScheduler::GlobalScheduler(const GlobalSchedulerParams &p)
    : DmaDevice(p), num_gpus(p.num_gpus), hsapp(p.hsapp),
      gpu_cmd_proc(p.device), dispatcher(p.dispatcher), driver(nullptr),
      n_cu(p.n_cu), n_wf(p.n_wf), num_SIMDs(p.num_SIMDs), wf_size(p.wf_size),
      pioAddr(p.pioAddr), pioDelay(p.pioDelay),
      policy(GSPolicyFactory::makePolicy(p.sched_policy)), num_sched_gpu(p.num_sched_gpu), default_acq_rel(p.default_acq_rel), num_tccs(p.num_tccs)
{
    of = std::ofstream(p.outdir+"/gs_con_test.txt");
    of << "event,tick,gpu,queue,kern_id,kern_name,kern_hash\n";
    of.flush();
    doorbell_reg = new uint32_t[512];
    kernelInfo = new KernelInfo(-1); // Replace with # kerns to hold
    cpcohTable = new CpCoh(100);
    availableWFs.resize(num_gpus);
    temp_buffer_size = 1024;
    GSDelay = 2000000;
    SendDelay = 100000;

    for (int i = 0; i < p.num_gpus; i++){
        availableWFs[i] = p.n_wf * p.num_SIMDs * p.n_cu;
        hsapp[i]->attachGlobalScheduler(this);
        hwSchdlr.push_back(hsapp[i]->hwSchdlr);
        hwSchdlr[i]->attachGlobalScheduler(this);
        gpu_cmd_proc[i]->attachGlobalScheduler(this);
        dispatcher[i]->attachGlobalScheduler(this);
    }
    gpu_cmd_proc[0]->system()->registerGlobalScheduler(this);
}

AddrRangeList
GlobalScheduler::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(RangeSize(pioAddr, 4096*num_gpus));
    return ranges;
}

Tick
GlobalScheduler::read(Packet *pkt)
{
    pkt->makeAtomicResponse();
    pkt->setBadAddress();
    return pioDelay;
}

Tick
GlobalScheduler::write(Packet *pkt)
{
    Addr db_offset = (Addr) ((pkt->getAddr() - pioAddr) % 4096);
    uint32_t queue_id = dbMapGlobal[db_offset];
    uint32_t num_kernels = pkt->getLE<uint32_t>() - qInfo[queue_id]->dmaIdx;
    assert(num_kernels <= qInfo[queue_id]->aqlBuf->numObjs());

    DPRINTF(HSAPacketProcessor,
                "db_offset = 0x%x queue_id = %d num_kernels = %d\n",
                db_offset, queue_id, num_kernels);
    qInfo[queue_id]->aqlBuf->allocEntry(num_kernels);
    assert(!qInfo[queue_id]->qDesc->dmaInProgress);
    qInfo[queue_id]->qDesc->dmaInProgress = true;
    qInfo[queue_id]->kernelsDmaing = num_kernels;

    int kernels_to_dma = num_kernels;
    doorbell_reg[queue_id] = pkt->getLE<uint32_t>();
    while (kernels_to_dma > 0) {
        int space = qInfo[queue_id]->aqlBuf->numObjs() -
            (qInfo[queue_id]->dmaIdx % qInfo[queue_id]->aqlBuf->numObjs());
        space = (space < kernels_to_dma) ? space : kernels_to_dma;
        void* aql_buf = qInfo[queue_id]->aqlBuf->ptr(qInfo[queue_id]->dmaIdx);

        Packet* pkt_pass = Packet::createWrite(pkt->req);
        pkt_pass->dataDynamic<uint32_t>(&(doorbell_reg[queue_id]));
        pkt_pass->setAddr(pkt->getAddr());
        DPRINTF(HSAPacketProcessor,
                    "num packets = %d old address = 0x%x new address = 0x%x\n",
                    space, pkt->getAddr(),
                    pkt_pass->getAddr());

        GSDmaEvent *dmaEvent = new GSDmaEvent(this, pkt_pass,
                                              queue_id, space, aql_buf);

        dmaReadVirt(qInfo[queue_id]->qDesc->ptr(qInfo[queue_id]->dmaIdx),
                    space * qInfo[queue_id]->qDesc->objSize(),
                    dmaEvent, aql_buf);

        qInfo[queue_id]->dmaIdx += space;
        kernels_to_dma -= space;
        //update read index here
    }
    pkt->makeAtomicResponse();
    return pioDelay;
}

void
GlobalScheduler::translateOrDie(Addr vaddr, Addr &paddr)
{
    auto process = sys->threads[0]->getProcessPtr();

    if (!process->pTable->translate(vaddr, paddr))
        fatal("failed translation: vaddr 0x%x\n", vaddr);
}

void
GlobalScheduler::dmaVirt(DmaFnPtr dmaFn, Addr addr, unsigned size,
                         Event *event, void *data, Tick delay)
{
    if (size == 0) {
        schedule(event, curTick() + delay);
        return;
    }

    // move the buffer data pointer with the chunks
    uint8_t *loc_data = (uint8_t*)data;

    for (ChunkGenerator gen(addr, size, PAGE_SIZE); !gen.done(); gen.next()) {
        Addr phys;
        // translate pages into their corresponding frames
        translateOrDie(gen.addr(), phys);

        Event *ev = gen.last() ? event : NULL;
        (this->*dmaFn)(phys, gen.size(), ev, loc_data, delay);
        loc_data += gen.size();
    }
}

void
GlobalScheduler::dmaReadVirt(Addr host_addr, unsigned size, Event *event,
                             void *data, Tick delay)
{
    DPRINTF(HSAPacketProcessor,
            "%s:host_addr = 0x%lx, size = %d\n", __FUNCTION__, host_addr,
            size);
    dmaVirt(&DmaDevice::dmaRead, host_addr, size, event, data, delay);
}

void
GlobalScheduler::dmaWriteVirt(Addr host_addr, unsigned size, Event *event,
                              void *data, Tick delay)
{
    DPRINTF(HSAPacketProcessor,
            "%s:host_addr = 0x%lx, size = %d\n", __FUNCTION__, host_addr,
            size);
    dmaVirt(&DmaDevice::dmaWrite, host_addr, size, event, data, delay);
}

/*void
GlobalScheduler::moveQueue(uint32_t queue_id, uint32_t gpu_id1,
                           uint32_t gpu_id2)
{
    // add queue_id to activeList gpu2
    DPRINTF(HSAPacketProcessor, "move queue %d from GPU %d to GPU %d\n",
                                queue_id, gpu_id1, gpu_id2);
    int from = gpu_id1 - STARTING_GPU_ID;
    int to = gpu_id2 - STARTING_GPU_ID;
    auto al_iter = hwSchdlr[from]->activeList.find(queue_id);
    if (al_iter != hwSchdlr[from]->activeList.end()) {
        auto rl_iter = hwSchdlr[from]->regdListMap.find(queue_id);
        if (rl_iter != hwSchdlr[from]->regdListMap.end()) {
            warn("queue already scheduled into registered list");
        }
        QCntxt q_cntxt = hwSchdlr[from]->activeList[queue_id];
        hwSchdlr[to]->registerQueue(q_cntxt, queue_id);
        hwSchdlr[from]->unregisterQueueForMove(queue_id);
        queue_to_gpu[queue_id] = gpu_id2;
    }
    else {
        panic("queue not found");
    }
}

void
GlobalScheduler::RRPlan(uint32_t queue_id, uint32_t num_kernels)
{
    lastGPU = ((num_kernels-1) % num_gpus) + STARTING_GPU_ID;
    for (int i = 0; i < num_kernels; i++) {
        gpu_seq_plan[queue_id].push_back(i % num_gpus + STARTING_GPU_ID);
    }
}*/


GlobalScheduler::GSDmaEvent::
GSDmaEvent(GlobalScheduler* _global_scheduler, Packet* _pkt_pass,
           uint32_t _queue_id, uint32_t _num_kernels, void* _data)
    : Event(Default_Pri, AutoDelete), glb_schdlr(_global_scheduler),
      pkt_pass(_pkt_pass), queue_id(_queue_id), num_kernels(_num_kernels),
      data(_data)
{
    setFlags(AutoDelete);
}

uint32_t GlobalScheduler::addKernelIdx = 0;

void
GlobalScheduler::GSDmaEvent::process()
{
    glb_schdlr->qInfo[queue_id]->kernelsDmaing -= num_kernels;
    glb_schdlr->qInfo[queue_id]->numKernels+=num_kernels;
    if (glb_schdlr->qInfo[queue_id]->kernelsDmaing == 0) {
        glb_schdlr->qInfo[queue_id]->qDesc->dmaInProgress = false;
        AQLRingBuffer *aqlRingBuffer M5_VAR_USED =
            glb_schdlr->qInfo[queue_id]->aqlBuf;
        bool isVendorPkt = false;
        uint32_t vendorPkt = 0;
        uint32_t curPkt = glb_schdlr->qInfo[queue_id]->pktIdx;

        for (; curPkt < glb_schdlr->qInfo[queue_id]->numKernels; curPkt++) {

            void* pkt_dma = aqlRingBuffer->ptr(curPkt);
            auto disp_pkt = (_hsa_dispatch_packet_t *)pkt_dma;
            hsa_packet_type_t pkt_type = PKT_TYPE(disp_pkt);

            if (pkt_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC){
                DPRINTF(HSAPacketProcessor, "Vendor Specific Packet\n");
                isVendorPkt = true;
                vendorPkt = curPkt;
                curPkt++;
                break;
            }
            else if (pkt_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
                 for(auto i = glb_schdlr->qInfo[queue_id]->kernels.begin(); i != glb_schdlr->qInfo[queue_id]->kernels.end(); i ++)
                 {
                    if(i->second.kernKey.kernelName.length() <= 1){
                      glb_schdlr->qInfo[queue_id]->kernels.erase(i);  
                    }
                 }
          ///       if (IS_BARRIER(disp_pkt)){
         //           DPRINTF(GlobalScheduler, "Setting Barrier Bit");
          ///          glb_schdlr->qInfo[queue_id]->setBarrierBit(true);
           //     }    
                auto tc = glb_schdlr->sys->threads[0];
                auto &virt_proxy = tc->getVirtProxy();

                AMDKernelCode akc;
                virt_proxy.readBlob(disp_pkt->kernel_object, (uint8_t*)&akc,
                    sizeof(AMDKernelCode));

                Addr kern_name_addr(0);
                std::string kernel_name;

                if (akc.runtime_loader_kernel_symbol) {
                    virt_proxy.readBlob(
                        akc.runtime_loader_kernel_symbol + 0x10,
                        (uint8_t*)&kern_name_addr, 0x8);

                    virt_proxy.readString(kernel_name, kern_name_addr);

                    DPRINTF(GlobalScheduler, "Adding kernel %s with ID %d to "
                                "queue %d\n", kernel_name, curPkt, queue_id);
                                

                    std::hash<std::string> str_hash;
                    size_t kern_hash = str_hash(kernel_name+
                        std::to_string(disp_pkt->workgroup_size_x)+
                        std::to_string(disp_pkt->workgroup_size_y)+
                        std::to_string(disp_pkt->workgroup_size_z));

                    glb_schdlr->recordSchedulingEvent(EVENTS::ENQ, queue_id, 0,
                        curPkt, kernel_name, kern_hash);
                    auto total_wgs = divCeil(disp_pkt->grid_size_x, disp_pkt->workgroup_size_x)*divCeil(disp_pkt->grid_size_y, disp_pkt->workgroup_size_y)*divCeil(disp_pkt->grid_size_z, disp_pkt->workgroup_size_z);
                    glb_schdlr->num_sched_gpu = (total_wgs < glb_schdlr->num_sched_gpu) ? total_wgs : glb_schdlr->num_sched_gpu;   
                    glb_schdlr->kernelInfo->addKernel(kernel_name, disp_pkt, glb_schdlr->num_sched_gpu);
                    glb_schdlr->qInfo[queue_id]->addKernel(kernel_name,
                                                           disp_pkt,
                                                           curPkt, glb_schdlr->num_sched_gpu);
                    GlobalScheduler::addKernelIdx++;
                } else {
                    kernel_name = "Blit kernel";
                }

                DPRINTF(HSAPacketProcessor, "Kernel name: %s\n",
                                            kernel_name.c_str());

                DPRINTF(HSAPacketProcessor, "Got AQL: wg size (%dx%dx%d), "
                    "grid size (%dx%dx%d) kernarg addr: %#x, completion "
                    "signal addr:%#x, private segment: %#x,group segment: %#x,"
                    "kernel object: %#x, reserved 0: %#x, reserved 1: %#x\n",
                    disp_pkt->workgroup_size_x,
                    disp_pkt->workgroup_size_y, disp_pkt->workgroup_size_z,
                    disp_pkt->grid_size_x, disp_pkt->grid_size_y,
                    disp_pkt->grid_size_z, disp_pkt->kernarg_address,
                    disp_pkt->completion_signal,
                    disp_pkt->private_segment_size,
                    disp_pkt->group_segment_size, disp_pkt->kernel_object,
                    disp_pkt->reserved0, disp_pkt->reserved1);
            } else if (pkt_type == HSA_PACKET_TYPE_BARRIER_AND) {
                for(auto i = glb_schdlr->qInfo[queue_id]->kernels.begin(); i != glb_schdlr->qInfo[queue_id]->kernels.end(); i ++)
                 {
                    if(i->second.kernKey.kernelName.length() <= 1){
                      glb_schdlr->qInfo[queue_id]->kernels.erase(i);  
                    }
                 } 
                //assert(!glb_schdlr->qInfo[queue_id]->hasBarrier());
                DPRINTF(GlobalScheduler, "Assigning barr to Q[%d] idx[%d]\n",
                        queue_id, curPkt);
                auto barrPkt = (_hsa_barrier_and_packet_t *)pkt_dma;
                glb_schdlr->qInfo[queue_id]->setBarrier(barrPkt, curPkt);
            }
        }
        glb_schdlr->qInfo[queue_id]->pktIdx = curPkt;

        uint32_t doorbell_reg = pkt_pass->getLE<uint32_t>();
        assert(doorbell_reg <= 1024);

        //make scheduling decision
        DPRINTF(GlobalScheduler, "Got a new pkt and will make a new scheduling decision\n");
        fflush(stdout);
        glb_schdlr->makeSchedulingDecision(queue_id, isVendorPkt, false,
                                           vendorPkt);
    }
}

void
GlobalScheduler::kernelComplete(uint32_t queue_id, uint32_t kern_id)
{
    DPRINTF(GlobalScheduler, "Kernel %d from queue %d complete.\n",
            kern_id, queue_id);
    DPRINTF(GlobalScheduler, "Queue %d has %d kernels\n",
            queue_id, qInfo[queue_id]->numKernels);
    GSReceiveEvent* event = new GSReceiveEvent(this, queue_id, kern_id);
    schedule(event, curTick() + SendDelay);
}

void
GlobalScheduler::makeSchedulingDecision(uint32_t queue_id, bool isVendorPkt,
                                        bool global, uint32_t kern_id)
{
    if (qInfo[queue_id]->scheduled || (qInfo[queue_id]->numDispatched() > 0 && !isVendorPkt)) {
        DPRINTF(GlobalScheduler, "Already scheduled scheduling for queue %d, "
                "returning\n", queue_id);
        //processBarrier(queue_id);        
        recordSchedulingEvent(EVENTS::FAIL_QUEUE, queue_id);
        return;
    }
    GSDecideKernelEvent* event = new GSDecideKernelEvent(this, queue_id,
                                                         isVendorPkt,
                                                         kern_id);
    Tick delay = GSDelay;
    // If we get a schedule request from dispatcher, want to add
    // SendDelay because we need to account for the GPU->Scheduler
    // transmission time
    if (global) {
        delay += SendDelay;
    }
    qInfo[queue_id]->scheduled = true;
    DPRINTF(GlobalScheduler, "Queue[%d]: Scheduling scheduling at tick %d and scheduled is %d\n",
            queue_id, curTick() + delay, qInfo[queue_id]->scheduled );

    recordSchedulingEvent(EVENTS::SCHED);

    schedule(event, curTick() + delay);
}

GlobalScheduler::GSReceiveEvent::
GSReceiveEvent(GlobalScheduler* _global_scheduler, uint32_t _queue_id,
               uint32_t _kern_id)
    : Event(Default_Pri, AutoDelete), glb_schdlr(_global_scheduler),
      queue_id(_queue_id), kern_id(_kern_id)
{
    setFlags(AutoDelete);
}

void
GlobalScheduler::GSReceiveEvent::process()
{
    if (glb_schdlr->qInfo[queue_id]->isDispatchedKern(kern_id)) {
        uint32_t gpu_id = glb_schdlr->qInfo[queue_id]->
                            getKernelGPU(kern_id) - STARTING_GPU_ID;
        DPRINTF(GlobalScheduler, "GPU %d now has %d WFs (was %d)\n",
                gpu_id, int32_t(glb_schdlr->availableWFs[gpu_id]+
                glb_schdlr->kernelWFs(queue_id, kern_id)),
                glb_schdlr->availableWFs[gpu_id]);
        fflush(stdout);
        fflush(stdout);
        KernelKey kk = glb_schdlr->qInfo[queue_id]->getKernKey(kern_id);
        glb_schdlr->recordSchedulingEvent(EVENTS::COMP, queue_id, gpu_id,
            kern_id, kk.kernelName, kk.hash);
        fflush(stdout);
        glb_schdlr->availableWFs[gpu_id] +=
            glb_schdlr->kernelWFs(queue_id, kern_id);
        glb_schdlr->updateRuntimeEstimate(queue_id, kern_id);
        glb_schdlr->qInfo[queue_id]->completeKernel(kern_id, gpu_id + STARTING_GPU_ID);
         DPRINTF(GlobalScheduler, "Number of Chiplets still to finish for this kernel are  %d\n", glb_schdlr->qInfo[queue_id]->dispKernels[kern_id].chiplets.size());
        
        
        bool currentlyExecuting = false;
        
        for(auto i = glb_schdlr->qInfo[queue_id]->dispKernels.begin(); i != glb_schdlr->qInfo[queue_id]->dispKernels.end(); i ++)
                 {
                    if(i->second.kernKey.kernelName.length() <= 1){
                      glb_schdlr->qInfo[queue_id]->dispKernels.erase(i);  
                    }
                 } 

        if(glb_schdlr->qInfo[queue_id]->dispKernels.size() == 0){
            glb_schdlr->processBarrier(queue_id);
        }
        for (auto &q : glb_schdlr->qInfo) {
            if (q.second->numDispatched() > 0 && q.first == queue_id && glb_schdlr->qInfo[queue_id]->dispKernels[glb_schdlr->qInfo[queue_id]->dispatchedKernIdx()].kernKey.kernelName.length() > 1) {
                currentlyExecuting = true;
                break;
            }
        }
        if (!currentlyExecuting || !glb_schdlr->qInfo[queue_id]->scheduled) {
            glb_schdlr->makeSchedulingDecision(queue_id, false, false);
        } else {
            glb_schdlr->recordSchedulingEvent(EVENTS::FAIL_QUEUE, queue_id);
        }
    } else {
        DPRINTF(GlobalScheduler, "Returned kernel (%d) wasn't kernel "
                "dispatch packet.\n", kern_id);
    }
}

GlobalScheduler::GSSendKernelEvent::
GSSendKernelEvent(GlobalScheduler* _global_scheduler, uint32_t _gpu_id,
                  uint32_t _queue_id, uint32_t _writeIndex,
                  uint32_t _readIndex, uint32_t _dispIndex, bool isVendor)
    : Event(Default_Pri, AutoDelete), glb_schdlr(_global_scheduler),
      gpu_id(_gpu_id), queue_id(_queue_id), writeIndex(_writeIndex),
      readIndex(_readIndex), dispIndex(_dispIndex), isVendor(isVendor)
{
    setFlags(AutoDelete);
}

void
GlobalScheduler::GSSendKernelEvent::process()
{
    glb_schdlr->recordSchedulingEvent(EVENTS::DISP, queue_id,
                                      gpu_id - STARTING_GPU_ID, dispIndex);

    DPRINTF(GlobalScheduler, "Number of chiplets this kernel will be scheduled on is  %d\n", glb_schdlr->qInfo[queue_id]->dispKernels[readIndex].chiplets.size());
    fflush(stdout);
    if (isVendor)
    {
        glb_schdlr->hsapp[gpu_id - STARTING_GPU_ID]->write_packet(queue_id,
                                                                  writeIndex, readIndex, dispIndex);
    }
    else
    {
         glb_schdlr->prepareCPCohArgs(glb_schdlr->qInfo[queue_id]->dispKernels[readIndex].chiplets, queue_id, readIndex, addKernelIdx); //CPCOH needs to know the dispatch index as well
        for (auto i = glb_schdlr->qInfo[queue_id]->dispKernels[readIndex].chiplets.begin(); i != glb_schdlr->qInfo[queue_id]->dispKernels[readIndex].chiplets.end(); i++)
        {
            glb_schdlr->hsapp[*i - STARTING_GPU_ID]->write_packet(queue_id,
                                                                  writeIndex, readIndex, dispIndex);
        }
    }
   // }
}

GlobalScheduler::GSDecideKernelEvent::
GSDecideKernelEvent(GlobalScheduler* _global_scheduler, uint32_t _queue_id,
                    bool _isVendorPkt, uint32_t _vendor_id)
    : Event(Default_Pri, AutoDelete), glb_schdlr(_global_scheduler),
      queue_id(_queue_id), isVendorPkt(_isVendorPkt), vendor_id(_vendor_id)
{
    setFlags(AutoDelete);
}

void
GlobalScheduler::GSDecideKernelEvent::process()
{
    if (isVendorPkt) {
        glb_schdlr->sendVendor(queue_id, vendor_id);
    }
    else {
        glb_schdlr->sendKernel(queue_id);
    }
    glb_schdlr->qInfo[queue_id]->scheduled = false;
    DPRINTF(GlobalScheduler, "Schedule set to false\n");

}

void
GlobalScheduler::sendVendor(uint32_t queue_id, uint32_t pkt_id)
{
    uint32_t gpu_id = qInfo[queue_id]->dispGpu;
    uint32_t writeIndex = doorbell_reg[queue_id];
    uint32_t readIndex = pkt_id;
    uint32_t dispIndex = pkt_id;
    bool isVendor = true;
    GSSendKernelEvent* event = new GSSendKernelEvent(this, gpu_id, queue_id,
        writeIndex, readIndex, dispIndex, isVendor);
    schedule(event, curTick() + SendDelay);
}

void
GlobalScheduler::sendKernel(uint32_t queue_id)
{
    Event *event = policy->chooseKernel(this, queue_id);
    if (event)
    {
        DPRINTF(GlobalScheduler, "Queue[%d]: Scheduling kernel launch\n",
                queue_id);
        schedule(event, curTick() + SendDelay);
    } else {
        DPRINTF(GlobalScheduler, "Queue[%d]: Nothing to schedule\n", queue_id);
        processBarrier(queue_id);
    }
}

void GlobalScheduler::prepareCPCohArgs(std::set<uint32_t> chiplets, uint32_t queue_id, uint32_t kernel_id, uint32_t cpcoh_dispKernIdx)
{
    chipletVector schedV;
    for (auto i = chiplets.begin(); i != chiplets.end(); i++)
    {
        schedV[*i - STARTING_GPU_ID] = bitset<2>(1);
        //Creating the scheduling wherever the chiplet is scheduled is marked as 01
    }
    if(default_acq_rel){
    chipletID FlushVec = std::bitset<NUM_CHIPLET>{0xf};
    qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control = std::make_pair(FlushVec, FlushVec);
    setChipletInvalidate(FlushVec , schedV, queue_id, kernel_id); 
    setChipletFlush(FlushVec, schedV, queue_id, kernel_id);
    }
    else if(incomingKernelArgs[cpcoh_dispKernIdx - 1].empty()){
        chipletID FlushVec = std::bitset<NUM_CHIPLET>{0x0};
        qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control = std::make_pair(FlushVec, FlushVec);
        setChipletInvalidate(FlushVec , schedV, queue_id, kernel_id); 
        setChipletFlush(FlushVec, schedV, queue_id, kernel_id);
    }
    else {    
    auto ArgMap = incomingKernelArgs[cpcoh_dispKernIdx - 1];    //Now incomingKernelArgs' a vector
    schedulerVector CpCohVec;
    for (auto ArgVec : ArgMap)
    {
        chipletVector modeV;
        for (auto i = 0; i < schedV.size(); i++)
        {
            modeV[i] =   (schedV[i] == 1) ? get<1>(ArgVec) : 0;
            // We are defining Read Only as 0x0 and R/W as 0x3
        }
        CpCohVec.push_back(std::make_tuple(get<0>(ArgVec), schedV, modeV, get<2>(ArgVec)));
    }
    std::pair<chipletID, chipletID> CPCoh_queues = cpcohTable->putcpcohEntry(CpCohVec);
    qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control = CPCoh_queues;
    setChipletInvalidate(CPCoh_queues.first, schedV, queue_id, kernel_id); 
    setChipletFlush(CPCoh_queues.second, schedV, queue_id, kernel_id);
        //std::cout << CpCoh_queues.first; // Just to prevent WError
    //Rajesh Q: do you need number of Args? just do CpCohVec.size()
    //incomingKernelArgs.pop_back();
  }
}

void
GlobalScheduler::createQueue(uint64_t hostReadIndexPointer,
                              uint64_t basePointer,
                              uint64_t queue_id,
                              uint32_t size,
                              uint32_t gpu_id)
{
    assert(queue_id < MAX_ACTIVE_QUEUES);
    // Map queue ID to doorbell.
    // We are only using offset to pio base address as doorbell
    // We use the same mapping function used by hsa runtime to do this mapping
    Addr db_offset = sizeof(uint32_t)*queue_id;
    if (dbMapGlobal.find(db_offset) != dbMapGlobal.end()) {
        panic("Creating an already existing queue (queueID %d)", queue_id);
    }

    // Populate global doorbell map
    dbMapGlobal[db_offset] = queue_id;

    if (queue_id >= MAX_ACTIVE_QUEUES) {
        panic("Attempting to create a queue (queueID %d)" \
              " beyond PIO range", queue_id);
    }

    qInfo[queue_id] = new QInfo(basePointer, db_offset,
                                hostReadIndexPointer,
                                size, gpu_id);

    for (int i = 0; i < num_gpus; i++) {
        hsapp[i]->registerQueueFromGlobal(hostReadIndexPointer,
            basePointer,
            queue_id,
            size);
    }

}

void
GlobalScheduler::destroyQueue(uint32_t queue_id) {
    for (int i = 0; i < num_gpus; i++) {
        hsapp[i]->unsetDeviceQueueDesc(queue_id);
    }
}

void
GlobalScheduler::updateQueue(uint32_t queue_id, uint32_t queue_priority) {
    DPRINTF(GlobalScheduler, "Queue[%d]: Update priority to %d\n",
            queue_id, queue_priority);
    qInfo[queue_id]->setPriority(queue_priority);
    for (int i = 0; i < num_gpus; i++) {
        hwSchdlr[i]->updateQueuePriority(queue_id, queue_priority);
    }
}

uint32_t
GlobalScheduler::kernelWFs(uint32_t queue_id, uint32_t kern_id)
{
    return (qInfo[queue_id]->getKernelWGs(kern_id) *
            qInfo[queue_id]->getKernelThreads(kern_id) + wf_size- 1)
           / wf_size;
}

uint32_t
GlobalScheduler::kernelWGSize(uint32_t queue_id, uint32_t kern_id)
{
    return (qInfo[queue_id]->getKernelThreads(kern_id) + wf_size - 1)
           / wf_size;
}

void
GlobalScheduler::markKernsForDisp(uint32_t queue_id, uint32_t kern_id_start,
                                  uint32_t kern_id_end, uint32_t gpu_id, bool isCopy)
{
    if (!qInfo[queue_id]->jobStarted() && ! isCopy) {
        qInfo[queue_id]->setJobStart();
    }

    for (int i = kern_id_start; i < kern_id_end; i++)
    {
        if (qInfo[queue_id]->isKern(i))
        {
            DPRINTF(GlobalScheduler, "Scheduling Q[%d] Kern[%d](%s) to gpu[%d]"
                    " WFs needed: %d. (WFs before scheduling: %d)\n",
                    queue_id, i, qInfo[queue_id]->getKernKey(i).kernelName,
                    gpu_id - STARTING_GPU_ID,
                    kernelWFs(queue_id, i),
                    availableWFs[gpu_id - STARTING_GPU_ID]);
            if(!isCopy) {qInfo[queue_id]->dispGpu = gpu_id;} //Change required here
            qInfo[queue_id]->markKernForDisp(i, gpu_id);
            availableWFs[gpu_id - STARTING_GPU_ID] -= kernelWFs(queue_id, i);
        } else {
            DPRINTF(GlobalScheduler, "Non-kernel dispatch %d on queue %d "
                    "scheduled to gpu %d\n", i, queue_id,
                    gpu_id - STARTING_GPU_ID);
        }
    }
}

void
GlobalScheduler::markKernDispatched(uint32_t queue_id, uint32_t kern_id)
{
    DPRINTF(GlobalScheduler, "Queue[%d]: Marking Kernel[%d] as dispatched\n",
            queue_id, kern_id);

    qInfo[queue_id]->markKernDispatched(kern_id);
}

void
GlobalScheduler::kernelWgStart(uint32_t queue_id, uint32_t kern_id,
                               uint32_t wg_id)
{
    qInfo[queue_id]->startWGTime(kern_id, wg_id);
}

void
GlobalScheduler::kernelWgFinish(uint32_t queue_id, uint32_t kern_id,
                                uint32_t wg_id)
{
    qInfo[queue_id]->finishWGTime(kern_id, wg_id);
}

void
GlobalScheduler::updateRuntimeEstimate(uint32_t queue_id, uint32_t kern_id)
{
    kernelInfo->updateTime(qInfo[queue_id]->getKernKey(kern_id),
                           qInfo[queue_id]->getWGRuntimes(kern_id));
}

Tick
GlobalScheduler::estimateKernelRuntime(uint32_t queue_id, uint32_t kern_id,
                                       uint32_t gpu_id)
{
    uint32_t totWFs = qInfo[queue_id]->getKernelWGs(kern_id)
                      * kernelWFs(queue_id, kern_id);

    KernelKey key = qInfo[queue_id]->getKernKey(kern_id);

    // Uncomment when we want to use the estimated WFs
    //return (kernelInfo->getAvgTime(key) * totWFs) /
    //       std::min(totWFs, availableWFs[gpu_id]);

    return (kernelInfo->getAvgTime(key) * totWFs) /
           std::min(totWFs, wf_size);
}

void
GlobalScheduler::processBarrier(uint32_t queue_id)
{
    DPRINTF(GlobalScheduler, "Checking for barrier completion in Q[%d]\n",
            queue_id);
    for (int i = 0; i < qInfo[queue_id]->hasBarrier(); i++) {
        if (qInfo[queue_id]->checkBarrier(i)) {
            auto pkt = qInfo[queue_id]->getBarrier(i);
            bool isReady = true;

            for (int i = 0; i < NumSignalsPerBarrier; i++) {
                if (pkt->dep_signal[i]) {
                    DPRINTF(GlobalScheduler, "Barrier pkt not finished?\n");
                    isReady = false;
                    break;
                }
            }
            if (isReady) {
                DPRINTF(GlobalScheduler, "Barrier packet completed\n");
                if (pkt->completion_signal != 0) {
                    Addr value_addr = pkt->completion_signal +
                                      offsetof(amd_signal_t, value);
                    auto tc = sys->threads[0];
                    ConstVPtr<Addr> prev_value(value_addr, tc);
                    uint64_t old_signal_value = *prev_value;

                    Addr *new_signal = new Addr;
                    *new_signal = old_signal_value-1;

                    dmaWriteVirt(value_addr, sizeof(Addr), nullptr,
                                 new_signal, 0);

                    Addr mailbox_addr = pkt->completion_signal +
                                     offsetof(amd_signal_t, event_mailbox_ptr);
                    ConstVPtr<uint64_t> mailbox_ptr(mailbox_addr, tc);

                    if (*mailbox_ptr != 0) {
                        Addr event_addr = pkt->completion_signal +
                                          offsetof(amd_signal_t, event_id);

                        ConstVPtr<uint64_t> event_val(event_addr, tc);
                        driver->signalWakeupEvent(*event_val);
                    }
                }
                qInfo[queue_id]->clearBarrier(i);
                i--;
            }

        }
    }
}

void
GlobalScheduler::attachDriver(HSADriver *hsa_driver)
{
    fatal_if(driver, "Should not overwrite driver.");
    driver = hsa_driver;
}

bool
GlobalScheduler::lastDispToGpu(uint32_t queue_id, HSAPacketProcessor *pp)
{
    auto it = std::find(hsapp.begin(), hsapp.end(), pp);
    if (it != hsapp.end()) {
        uint32_t gpu_id = std::distance(hsapp.begin(), it) + STARTING_GPU_ID;
        return qInfo[queue_id]->numDispToGpu(gpu_id) == 1;
    } else {
        fatal("Asking for HSAPP the scheduler doesn't have");
    }
}

void
GlobalScheduler::recordSchedulingEvent(EVENTS e, uint32_t queue_id,
    uint32_t gpu_id, uint32_t kern_id, std::string kern_name, size_t kern_hash)
{
    switch(e) {
        case EVENTS::ENQ: {
            of << "ENQ," << curTick() << ",," << queue_id << ",";
            of << kern_id << "," << kern_name << "," << kern_hash << "\n";
            break;
        }
        case EVENTS::DISP: {
            of << "DISP," << curTick() << "," << gpu_id << ",";
            of << queue_id << "," << kern_id << ",,\n";
            break;
        }
        case EVENTS::COMP: {
            of << "COMP," << curTick() << "," << gpu_id << ",";
            of << queue_id << "," << kern_id << "," << kern_name << ",";
            of << kern_hash << "\n";
            break;
        }
        case EVENTS::SCHED: {
            of << "SCHED," << curTick() << ",,,,,\n";
            break;
        }
        case EVENTS::GPU_REQ: {
            // Passed-in GPU ID isn't valid, need to manually find it
            uint32_t gpu_id_valid = qInfo[queue_id]->
                getKernelGPU(kern_id) - STARTING_GPU_ID;
            of << "GPU_REQ," << curTick() << "," << gpu_id_valid << ",";
            of << queue_id << "," << kern_id << ",,\n";
            break;
        }
        case EVENTS::FAIL_KERN: {
            of << "FAIL_KERN," << curTick() << ",,,,,\n";
            break;
        }
        case EVENTS::FAIL_GPU: {
            of << "FAIL_GPU," << curTick() << ",,,,,\n";
            break;
        }
        case EVENTS::FAIL_QUEUE: {
            of << "FAIL_QUEUE," << curTick() << ",," << queue_id << ",,,\n";
            break;
        }
    }
    of.flush();
}

void GlobalScheduler::setChipletInvalidate(chipletID invalidate_queue, chipletVector sv, int queue_id, int kernel_id)
{
    for (std::size_t i = 0; i < invalidate_queue.size(); ++i){
        if (invalidate_queue[i] && sv[i] != 1){
                //gpu_cmd_proc[i]->shader()->invL2 = 1; // need to call prepareFlush here when the schedule does not match the chiplet to be flushed
                auto cu_ptr = gpu_cmd_proc[i]->shader()->cuList[0];
                GPUDynInstPtr gpuDynInst = std::make_shared<GPUDynInst>(cu_ptr, nullptr,
                new KernelLaunchStaticInst(), cu_ptr->getAndIncSeqNum());               
                gpuDynInst->queue_id = queue_id;
                gpuDynInst->kern_id = kernel_id;
                //Do this in normal way as well
                auto req = std::make_shared<Request>(0, 0, 0,
                                         cu_ptr->requestorId(),
                                         0, -1);
                req->setCacheCoherenceFlags(Request::INV_L2);
                req->setNoKernelReq();
                cu_ptr->injectGlobalMemFence(gpuDynInst, true, req);
            }
        }
}

void GlobalScheduler::setChipletFlush(chipletID flush_queue, chipletVector sv, int queue_id, int kernel_id)
{
    // DPRINTF(GlobalScheduler, "Chiplet:%d, Flush:%d, Full value: %d\n",i, flush_queue[i], flush_queue);
    for (std::size_t i = 0; i < flush_queue.size(); ++i){
        if (flush_queue[i] && sv[i] != 1){
                //gpu_cmd_proc[i]->shader()->wbL2 = 1; // need to call prepareFlush here when the schedule does not match the chiplet to be flushed
                auto cu_ptr = gpu_cmd_proc[i]->shader()->cuList[0];
                GPUDynInstPtr gpuDynInst = std::make_shared<GPUDynInst>(cu_ptr, nullptr,
                new KernelLaunchStaticInst(), cu_ptr->getAndIncSeqNum());               
                gpuDynInst->queue_id = queue_id;
                gpuDynInst->kern_id = kernel_id;
                auto req = std::make_shared<Request>(0, 0, 0,
                                         cu_ptr->requestorId(),
                                         0, -1);
                req->setCacheCoherenceFlags(Request::FLUSH_L2);
                req->setNoKernelReq();
                cu_ptr->injectGlobalMemFence(gpuDynInst, true, req);
            }
        }
}

void GlobalScheduler::notifyMemSyncCompletion(int queue_id, int kernel_id, int chiplet_id, bool inv_or_wb)
{

    if (inv_or_wb)
    {
        qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control.first[chiplet_id] = 0;
    }
    else
    {
        qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control.second[chiplet_id] = 0;
        for(auto i = qInfo[queue_id]->dispKernels[kernel_id].chiplets.begin(); i != qInfo[queue_id]->dispKernels[kernel_id].chiplets.end(); i ++){
            dispatcher[*i - STARTING_GPU_ID]->scheduleDispatch(); 
        }
    }
}

bool GlobalScheduler::getInvalidateFlushControl(int chiplet_id, int kernel_id, int queue_id, bool inv_or_wb)
{

    if (inv_or_wb)
    {
        return qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control.first[chiplet_id];
    }
    else
    {
        return qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control.second[chiplet_id];
    }
}

bool GlobalScheduler::isInvL2Done(int kernel_id, int queue_id){
    return  (qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control.first == 0);  
}

bool GlobalScheduler::isFlushL2Done(int kernel_id, int queue_id){
    return  (qInfo[queue_id]->dispKernels[kernel_id].invalidate_flush_control.second == 0);  
}

int
GlobalScheduler::getHomeNode(Addr address, int gpu_id, int cu_id)
{
   if(homeNodeMap.count((address >> 12)) == 0){
        homeNodeMap[(address >> 12)] = (cu_id/(n_cu));
         DPRINTF(GlobalScheduler, "Setting home node for the addrres addr 0x%lx is %d with gpu_id %d and cu_id %d and n_cu is %d\n", ((address >> 12)), homeNodeMap[(address >> 12)], gpu_id, cu_id, n_cu);
        fflush(stdout);
   }
   return homeNodeMap[(address >> 12)];
}