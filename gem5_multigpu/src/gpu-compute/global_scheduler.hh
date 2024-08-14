#ifndef __GLOBAL_SCHEDULER_HH__
#define __GLOBAL_SCHEDULER_HH__

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <set>
#include <tuple>
#include <deque>

#include "base/intmath.hh"
#include "base/logging.hh"
#include "debug/GlobalScheduler.hh"
#include "dev/dma_device.hh"
#include "dev/hsa/hsa_packet.hh"
#include "dev/hsa/hsa_packet_processor.hh"
#include "gpu-compute/global_scheduling_policy.hh"
#include "gpu-compute/cpcoh.hh"
#include "mem/packet.hh"
#include "params/GlobalScheduler.hh"
#include "sim/sim_object.hh"

#define STARTING_GPU_ID 2765

struct GlobalSchedulerParams;
class HWScheduler;
class HSADriver;
class HSAPacketProcessor;
class GPUCommandProcessor;
class AQLRingBuffer;
class CpCoh;

typedef enum {
    ENQ = 0,
    DISP,
    COMP,
    SCHED,
    GPU_REQ,
    FAIL_KERN,
    FAIL_GPU,
    FAIL_QUEUE,
} EVENTS;


struct KernelKey {
    std::string kernelName;
    uint32_t wgSizeX;
    uint32_t wgSizeY;
    uint32_t wgSizeZ;
    size_t hash;

    bool operator<(const KernelKey& rhs) const
    {
        return std::tie(kernelName, wgSizeX, wgSizeY, wgSizeZ) <
               std::tie(rhs.kernelName, rhs.wgSizeX, rhs.wgSizeY, rhs.wgSizeZ);
    }

    KernelKey(std::string _kernelName, uint32_t _wgSizeX, uint32_t _wgSizeY,
              uint32_t _wgSizeZ)
        : kernelName(_kernelName), wgSizeX(_wgSizeX), wgSizeY(_wgSizeY),
          wgSizeZ(_wgSizeZ)
    {
        std::hash<std::string> str_hash;
        hash = str_hash(_kernelName+
                        std::to_string(wgSizeX)+
                        std::to_string(wgSizeY)+
                        std::to_string(wgSizeZ));
    }

    KernelKey(std::string _kernelName, _hsa_dispatch_packet_t *pkt)
        : kernelName(_kernelName)
    {
        wgSizeX = pkt->workgroup_size_x;
        wgSizeY = pkt->workgroup_size_y;
        wgSizeZ = pkt->workgroup_size_z;

        std::hash<std::string> str_hash;
        hash = str_hash(_kernelName+
                        std::to_string(wgSizeX)+
                        std::to_string(wgSizeY)+
                        std::to_string(wgSizeZ));
    }

    KernelKey() {}

};

typedef enum {
    NOT_DISP = 0,
    D,
    COMP_DISP,
} DISP_STATE;

struct KernelQInfo {
    uint32_t numWGs;
    uint32_t numThreads;
    std::vector<Tick> wgStartTime;
    std::vector<Tick> wgRunTime;
    DISP_STATE dispStatus;
    uint32_t dispGpu;
    std::set <uint32_t> chiplets;
    std::pair<chipletID, chipletID> invalidate_flush_control;
    uint32_t kernelNum;
    KernelKey kernKey;
    uint32_t numWFs;
    uint32_t num_gpus;
    uint32_t gpuDispatchedID;

    KernelQInfo(std::string _kernelName, _hsa_dispatch_packet_t *pkt,
              uint32_t _kernelNum, uint32_t num_gpus)
        : dispStatus(NOT_DISP), dispGpu(0), kernelNum(_kernelNum),
          kernKey(_kernelName, pkt), num_gpus(num_gpus), gpuDispatchedID(0)
    {
        numWGs = divCeil(pkt->grid_size_x, pkt->workgroup_size_x)
                 * divCeil(pkt->grid_size_y, pkt->workgroup_size_y)
                 * divCeil(pkt->grid_size_z, pkt->workgroup_size_z);
        numThreads = pkt->workgroup_size_x *
                     pkt->workgroup_size_y *
                     pkt->workgroup_size_z;
        wgStartTime = std::vector<Tick>(numWGs, 0);
        wgRunTime = std::vector<Tick>(numWGs, 0);
    }

    KernelQInfo() {}

};

struct WGTime {
    Tick averageTime;
    uint32_t numWGsExecuted;

    WGTime() : averageTime(0), numWGsExecuted(0)
    {}
    Tick fullTime()
    {
        return averageTime * numWGsExecuted;
    }
    void updateTime(Tick runtime, uint32_t wgsRan)
    {
        Tick newAverage = (fullTime() + runtime)/(numWGsExecuted+wgsRan);
        averageTime = newAverage;
        numWGsExecuted += wgsRan;
    }
};

class KernelInfo {
  public:
    KernelInfo(uint32_t _maxNumKerns) : maxNumKerns(_maxNumKerns) {}
    void addKernel(std::string kernelName, _hsa_dispatch_packet_t *pkt, int num_gpus)
    {
        if (kernTimeEst.find(KernelKey(kernelName, pkt)) == kernTimeEst.end())
        {
            DPRINTF(GlobalScheduler, "Inserting kernel %s to "
                    "kernel-runtime map.\n", kernelName);
            kernTimeEst[KernelKey(kernelName, pkt)] = WGTime();
        } else {
            DPRINTF(GlobalScheduler, "Kernel %s alredy in kernel-runtime"
                    " map.\n", kernelName);
        }
    }

    void updateTime(KernelKey kernel, std::vector<Tick> wgTimes)
    {
        DPRINTF(GlobalScheduler, "Updating kernel %s time estimate.\n",
                kernel.kernelName);
        if (kernTimeEst.find(kernel) != kernTimeEst.end())
        {
            Tick newRuntime = std::accumulate(wgTimes.begin(), wgTimes.end(),
                                              (Tick)0);
            Tick oldRuntime = kernTimeEst[kernel].averageTime;
            kernTimeEst[kernel].updateTime(newRuntime, wgTimes.size());
            DPRINTF(GlobalScheduler, "Updated avg time estimate from %d"
                    " to %d\n", oldRuntime,
                    kernTimeEst[kernel].averageTime);
        } else {
            DPRINTF(GlobalScheduler, "Tried to update kernel time estimate"
                    " for kernel not in map\n");
        }
    }

    Tick getAvgTime(KernelKey kernel)
    {
        auto it = kernTimeEst.find(kernel);
        if (it != kernTimeEst.end()) {
            return it->second.averageTime;
        } else {
            return 0;
        }
    }

  private:
    std::map<KernelKey, WGTime> kernTimeEst;
    uint32_t maxNumKerns;
};

struct BarrierPacket {
    _hsa_barrier_and_packet_t *barrier;
    uint32_t barrierIdx;
    BarrierPacket(_hsa_barrier_and_packet_t *b, uint32_t idx) :
        barrier(b), barrierIdx(idx) {}
};

class QInfo {
  public:
    // Should be monotonically increasing
    uint32_t numKernels;
    Tick jobStart;
    bool dispatched;
    uint32_t pktIdx;
    uint32_t dmaIdx;
    uint32_t dispGpu;
    uint32_t kernelsDmaing;
    HSAQueueDescriptor *qDesc;
    AQLRingBuffer *aqlBuf;
    // First element: Kernel ID
    std::map<uint32_t, KernelQInfo> kernels;
    std::map<uint32_t, KernelQInfo> dispKernels;


    std::vector<BarrierPacket> barriers;
    uint32_t priority;

    bool scheduled;

    QInfo(uint64_t basePointer, Addr db_offset, uint64_t hostReadIdxPtr,
          uint32_t size, uint32_t gpu_id)
        : numKernels(0), jobStart(0), dispatched(false),
          pktIdx(0), dmaIdx(0), dispGpu(gpu_id), kernelsDmaing(0),
          priority(7), scheduled(false)
    {
        qDesc = new HSAQueueDescriptor(basePointer, db_offset,
                                       hostReadIdxPtr, size);
        aqlBuf = new AQLRingBuffer(NUM_DMA_BUFS, "Temp buffer");
    }

    void addKernel(std::string kernelName, _hsa_dispatch_packet_t *pkt,
                   uint32_t kernIdx, uint32_t num_gpus)
    {
        DPRINTF(GlobalScheduler, "A kernel has been added with kernelName  %s\n", kernelName); 
        kernels[kernIdx] = KernelQInfo(kernelName, pkt, kernIdx, num_gpus);
    }

    bool hasKernels()
    {
        return !kernels.empty();
    }

    void markKernForDisp(uint32_t kernIdx, uint32_t gpuIdx)
    {
        kernels[kernIdx].chiplets.insert(gpuIdx);
        DPRINTF(GlobalScheduler, "Chiplet vector size is %d and num gpus is %d\n", kernels[kernIdx].chiplets.size(), kernels[kernIdx].num_gpus  );
        if(kernels[kernIdx].chiplets.size() == kernels[kernIdx].num_gpus)
        {
            dispKernels[kernIdx] = kernels[kernIdx];
            /*dispKernels[kernIdx].dispGpu = gpuIdx; */   dispKernels[kernIdx].dispGpu = *(kernels[kernIdx].chiplets.begin());
            dispKernels[kernIdx].chiplets = kernels[kernIdx].chiplets;
            DPRINTF(GlobalScheduler, "Kernel Erased and dispathched kernel KernKey is %s\n", dispKernels[kernIdx].kernKey.kernelName);

            kernels.erase(kernIdx);
            DPRINTF(GlobalScheduler, "Num Kernels is %d \n", kernels.size());
        }  


    }

    void markKernDispatched(uint32_t kernIdx)
    {
        dispKernels[kernIdx].dispStatus = D;
    }

    // Could also just change dispStatus
    void completeKernel(uint32_t kernIdx, uint32_t gpuIdx)
    {
        dispKernels[kernIdx].chiplets.erase(gpuIdx);
         DPRINTF(GlobalScheduler, "Erasing gpuIdx %d the new chiplets vector size is %d\n", gpuIdx, dispKernels[kernIdx].chiplets.size());
        if(dispKernels[kernIdx].chiplets.size() == 0){
            dispKernels.erase(kernIdx);
        }
        DPRINTF(GlobalScheduler, "dispKernels size is  %d\n", dispKernels.size());
    }

    uint32_t oldestKernIdx()
    {
        if (hasKernels())
            return kernels.begin()->first;
        else
            fatal("Tried to get oldest kernel when no kernels exist");
    }

    uint32_t newestKernIdx()
    {
        if (hasKernels())
            return kernels.rbegin()->first;
        else
            fatal("Tried to get newest kernel when no kernels exist");
    }

    uint32_t dispatchedKernIdx()
    {
        if (!dispKernels.empty()) {
            if (dispKernels.size() == 1) {
                return dispKernels.begin()->first;
            } else {
                warn("Multiple kernels are dispatched from a queue, "
                     "selecting the first\n");
                return dispKernels.begin()->first;
            }
        } else {
            fatal("Tried to get dispatched kernel when no kernels are "
                  "dispatched");
        }
    }

    bool isDispatchedKern(uint32_t kernIdx)
    {
        return (dispKernels.find(kernIdx) != dispKernels.end());
    }

    bool isKern(uint32_t kernIdx)
    {
        return (kernels.find(kernIdx) != kernels.end());
    }

    uint32_t getKernelGPU(uint32_t kernIdx)
    {
        if (isDispatchedKern(kernIdx))
            return *(dispKernels[kernIdx].chiplets.begin());
        else
            fatal("Tried to get GPU of kernel not dispatched");
    }

    uint32_t getKernelWGs(uint32_t kernIdx)
    {
        if (isKern(kernIdx))
            return kernels[kernIdx].numWGs;
        else if (isDispatchedKern(kernIdx))
            return dispKernels[kernIdx].numWGs;
        else
            fatal("Tried to get WGs of kernel that doesn't exist");
    }

    uint32_t getKernelThreads(uint32_t kernIdx)
    {
        if (isKern(kernIdx))
            return kernels[kernIdx].numThreads;
        else if (isDispatchedKern(kernIdx))
            return dispKernels[kernIdx].numThreads;
        else
            fatal("Tried to get threads of kernel that doesn't exist");
    }

    void startWGTime(uint32_t kernIdx, uint32_t wgIdx)
    {
        if (isDispatchedKern(kernIdx))
            if (wgIdx < dispKernels[kernIdx].numWGs)
                dispKernels[kernIdx].wgStartTime[wgIdx] = curTick();
            else
                fatal("Tried to start WG time for an out-of-bounds WG (%d)\n",
                      wgIdx);
        else
            warn_once("Tried to start WG time for a kernel (%d) "
                      "that doesn't exist\n", kernIdx);
    }

    void finishWGTime(uint32_t kernIdx, uint32_t wgIdx)
    {
        if (isDispatchedKern(kernIdx)) {
            if (wgIdx < dispKernels[kernIdx].numWGs) {
                Tick start = dispKernels[kernIdx].wgStartTime[wgIdx];
                dispKernels[kernIdx].wgRunTime[wgIdx] = curTick() - start;
            } else {
                fatal("Tried to stop a WG time for an out-of-bounds WG (%d)\n",
                      wgIdx);
            }
        } else {
           warn_once("Tried to stop WG time for a kernel (%d) "
                     "that doesn't exist\n", kernIdx);
        }
    }

    KernelKey getKernKey(uint32_t kernIdx)
    {
        if (isKern(kernIdx) && !isDispatchedKern(kernIdx)){
            DPRINTF(GlobalScheduler, " The kernel kernKey is %s and kernel size is %d ",  kernels[kernIdx].kernKey.kernelName, kernels.size());
            return kernels[kernIdx].kernKey;
        }
        else if (isDispatchedKern(kernIdx)){
            DPRINTF(GlobalScheduler, " The dispatched kernel kernKey is %s",  dispKernels[kernIdx].kernKey.kernelName);
            return dispKernels[kernIdx].kernKey;      
        }
        else
            fatal("Tried to get Kernel key for kernel that doesn't exist");
    }

    std::vector<Tick> getWGRuntimes(uint32_t kernIdx)
    {
        if (isDispatchedKern(kernIdx)) {
            return dispKernels[kernIdx].wgRunTime;
        } else {
            fatal("Tried to get runtimes for kernel that didn't exist or "
                  "hasn't run yet");
        }
    }

    uint32_t hasBarrier()
    {
        return barriers.size();
    }

    void setBarrier(_hsa_barrier_and_packet_t *pkt, uint32_t pktIdx)
    {
        barriers.push_back(BarrierPacket(pkt, pktIdx));
    }

    void clearBarrier(uint32_t idx)
    {
        barriers.erase(barriers.begin()+idx);
    }

    bool checkBarrier(uint32_t idx)
    {
        if (hasBarrier()) {
            bool unDispCheck = false;
            bool dispCheck = false;
            if (!kernels.empty()) {
                unDispCheck =
                    barriers.at(idx).barrierIdx < kernels.begin()->first;
                DPRINTF(GlobalScheduler, "unDispCheck: %d < %d?\n",
                        barriers.at(idx).barrierIdx, kernels.begin()->first);
            } else {
                unDispCheck = true;
            }

            if (!dispKernels.empty() && dispKernels.begin()->second.kernKey.kernelName.length() > 1) {
                dispCheck =
                    barriers.at(idx).barrierIdx < dispKernels.begin()->first;
                DPRINTF(GlobalScheduler, "dispCheck: %d < %d?\n",
                        barriers.at(idx).barrierIdx,
                        dispKernels.begin()->first);
            } else {
                dispCheck = true;
            }

            return unDispCheck && dispCheck;
        }
        return false;
    }

    _hsa_barrier_and_packet_t *getBarrier(uint32_t idx)
    {
        return barriers.at(idx).barrier;
    }

    uint32_t numDispToGpu(uint32_t gpu_id)
    {
        auto func = [&](int sum, auto kern) {
            if (kern.second.dispGpu == gpu_id)
                sum += 1;
            return sum;
        };

        return std::accumulate(dispKernels.begin(), dispKernels.end(), 0,
                               func);
    }

    bool jobStarted()
    {
        return dispatched;
    }

    uint32_t numDispatched()
    {
        return dispKernels.size();
    }

    void setJobStart()
    {
        assert(!dispatched);
        dispatched = true;
        jobStart = curTick();
    }

    Tick getJobStart() {
        return jobStart;
    }

    std::vector<uint32_t> getKernelIdxs()
    {
        std::vector<uint32_t> kernIdxs;
        for (const auto& kerns : kernels)
            kernIdxs.push_back(kerns.first);
        return kernIdxs;
    }

    void setPriority(uint32_t newPriority)
    {
        priority = newPriority;
    }

    uint32_t getPriority()
    {
        return priority;
    }
};

class GlobalScheduler : public DmaDevice
{
  protected:
    typedef void (DmaDevice::*DmaFnPtr)(Addr, int, Event*, uint8_t*, Tick);
    void dmaVirt(DmaFnPtr, Addr host_addr, unsigned size,
                 Event* event, void *data, Tick delay = 0);
    void dmaReadVirt(Addr host_addr, unsigned size, Event* event,
                     void *data, Tick delay = 0);
    void dmaWriteVirt(Addr host_addr, unsigned size, Event* event,
                      void *data, Tick delay = 0);
    void translateOrDie(Addr vaddr, Addr &paddr);
    virtual Tick read(Packet *pkt);
    virtual Tick write(Packet *pkt);
    virtual AddrRangeList getAddrRanges() const;

  public:
    typedef GlobalSchedulerParams Params;
    GlobalScheduler(const Params &p);

    //void moveQueue(uint32_t queue_id, uint32_t gpu_id1, uint32_t gpu_id2);
    //void RRPlan(uint32_t queue_id, uint32_t num_packets);
    void createQueue(uint64_t hostReadIndexPointer,
                     uint64_t basePointer,
                     uint64_t queue_id,
                     uint32_t size,
                     uint32_t gpu_id);
    void destroyQueue(uint32_t queue_id);
    void updateQueue(uint32_t queue_id, uint32_t queue_priority);
    void sendVendor(uint32_t queue_id, uint32_t pkt_id);
    void sendKernel(uint32_t queue_id);
    void makeSchedulingDecision(uint32_t queue_id, bool isVendorPkt,
                                bool global, uint32_t kern_id = 0);
    void kernelComplete(uint32_t queue_id, uint32_t kern_id);

    void markKernsForDisp(uint32_t queue_id, uint32_t kern_id_start,
                          uint32_t kern_id_end, uint32_t gpu_id, bool isCopy = false);
    void markKernDispatched(uint32_t queue_id, uint32_t kern_id);
    void kernelWgStart(uint32_t queue_id, uint32_t kern_id, uint32_t wg_id);
    void kernelWgFinish(uint32_t queue_id, uint32_t kern_id, uint32_t wg_id);
    void updateRuntimeEstimate(uint32_t queue_id, uint32_t kern_id);
    Tick estimateKernelRuntime(uint32_t queue_id, uint32_t kern_id,
                               uint32_t gpu_id);
    void processBarrier(uint32_t queue_id);
    void attachDriver(HSADriver *hsa_driver);
    bool lastDispToGpu(uint32_t queue_id, HSAPacketProcessor *pp);
    // TODO: Update to use GPU id when GPUs have different configs
    uint32_t kernelWFs(uint32_t queue_id, uint32_t kern_id);
    uint32_t kernelWGSize(uint32_t queue_id, uint32_t kern_id);

    void recordSchedulingEvent(EVENTS e, uint32_t queue_id = 0,
                               uint32_t gpu_id = 0, uint32_t kern_id = 0,
                               std::string kern_name = {},
                               size_t kern_hash = 0);
    void prepareCPCohArgs(std::set <uint32_t> chiplets, uint32_t queue_id, uint32_t kern_idx, uint32_t cpcoh_dispKernIdx);
    std::map<Addr, uint32_t> dbMapGlobal;
    void setChipletInvalidate(chipletID invalidate_queue, chipletVector cv, int queue_id, int kernel_id);
    void setChipletFlush(chipletID flush_queue, chipletVector cv, int queue_id, int kernel_id);
    void notifyMemSyncCompletion(int queue_id, int kernel_id, int chiplet_id , bool inv_or_wb);
    bool getInvalidateFlushControl(int chiplet_id, int kernel_id, int queue_id, bool inv_or_wb);
    bool isInvL2Done(int kernel_id, int queue_id);
    bool isFlushL2Done(int kernel_id, int queue_id);
    int getHomeNode(Addr address, int gpu_id, int cu_id);

    //Scheduling Information
    std::vector<int32_t> availableWFs;
    std::map<uint32_t, QInfo*> qInfo;
    KernelInfo *kernelInfo;
    std::map<uint32_t, std::vector<uint32_t>> gpu_seq_plan;
    uint32_t num_gpus;

    uint32_t lastGPU;
    uint32_t* doorbell_reg;
    std::vector<std::vector<std::tuple<Addr,std::bitset<2>, bool>>> incomingKernelArgs = std::vector<std::vector<std::tuple<Addr,std::bitset<2>, bool>>> (100);
    CpCoh *cpcohTable;
    static uint32_t addKernelIdx;

    class GSDmaEvent : public Event
    {
      protected:
        GlobalScheduler* glb_schdlr;
        Packet* pkt_pass;
        uint32_t queue_id;
        uint32_t num_kernels;
        void* data;

      public:
        GSDmaEvent(GlobalScheduler *_global_scheduler, Packet *_pkt_pass,
                   uint32_t _queue_id, uint32_t _num_kernels, void* _data);
        virtual void process();
        virtual const char *description() const;
    };

    class GSSendKernelEvent : public Event
    {
      protected:
        GlobalScheduler* glb_schdlr;
        uint32_t gpu_id;
        uint32_t queue_id;
        uint32_t writeIndex;
        uint32_t readIndex;
        uint32_t dispIndex;
        bool isVendor;

      public:
        GSSendKernelEvent(GlobalScheduler* _global_scheduler, uint32_t _gpu_id,
                          uint32_t _queue_id, uint32_t _writeIndex,
                          uint32_t _readIndex, uint32_t _dispIndex, bool isVendor);
        virtual void process();
        virtual const char *description() const;
    };

    class GSDecideKernelEvent : public Event
    {
      protected:
        GlobalScheduler* glb_schdlr;
        uint32_t queue_id;
        bool isVendorPkt;
        uint32_t vendor_id;

      public:
        GSDecideKernelEvent(GlobalScheduler* _global_scheduler,
                            uint32_t _queue_id, bool _isVendorPkt,
                            uint32_t _vendor_id);
        virtual void process();
        virtual const char *description() const;
    };

    class GSReceiveEvent : public Event
    {
      protected:
        GlobalScheduler* glb_schdlr;
        uint32_t queue_id;
        uint32_t kern_id;

      public:
        GSReceiveEvent(GlobalScheduler* _global_scheduler, uint32_t _queue_id,
                       uint32_t _kern_id);
        virtual void process();
        virtual const char *description() const;
    };

  private:
    std::vector<HSAPacketProcessor*> hsapp;
    std::vector<GPUCommandProcessor*> gpu_cmd_proc;
    std::vector<GPUDispatcher*> dispatcher;
    std::vector<HWScheduler*> hwSchdlr;
    std::unordered_map<Addr, int> homeNodeMap;
    HSADriver *driver;

    uint32_t n_cu;
    uint32_t n_wf;
    uint32_t num_SIMDs;
    uint32_t temp_buffer_size;
    uint32_t wf_size;
    Addr pioAddr;
    Tick pioDelay;

    Tick GSDelay;
    Tick SendDelay;
    GSPolicy *policy;
    std::ofstream of;
    uint32_t num_sched_gpu;
    bool default_acq_rel;
    uint32_t num_tccs;

};

#endif
