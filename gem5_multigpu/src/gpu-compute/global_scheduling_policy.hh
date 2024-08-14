#ifndef __GPU_COMPUTE_GLOBAL_SCHEDULING_POLICY_HH__
#define __GPU_COMPUTE_GLOBAL_SCHEDULING_POLICY_HH__

#include <string>
#include <vector>

#include "base/logging.hh"
#include "enums/GSPolicyType.hh"

#define STARTING_GPU_ID 2765

class Event;
class GlobalScheduler;

/**
 * Interface class for the global scheduling policy.
 */
class GSPolicy
{
  public:
    GSPolicy() : nextGPU(STARTING_GPU_ID) { }

    virtual Event*
    chooseKernel(GlobalScheduler *gs, uint32_t qID) = 0;

  protected:
    uint32_t nextGPU;
    uint32_t readIdx, writeIdx;
    uint32_t gpu_id;
};

class RRGlobalPolicy : public GSPolicy
{
  public:
    RRGlobalPolicy() { }

    Event* chooseKernel(GlobalScheduler *gs, uint32_t qID) override;
};

class NormalGlobalPolicy : public GSPolicy
{
  public:
    NormalGlobalPolicy() { }

    Event* chooseKernel(GlobalScheduler *gs, uint32_t qID) override;

};

class LaxityGlobalPolicy : public GSPolicy
{
  public:
    LaxityGlobalPolicy() { }

    Event* chooseKernel(GlobalScheduler *gs, uint32_t qID) override;
};

class RRCSGlobalPolicy : public GSPolicy
{
  public:
    RRCSGlobalPolicy() { }

    Event* chooseKernel(GlobalScheduler *gs, uint32_t qID) override;
};

class GSPolicyFactory
{
  public:
    static GSPolicy* makePolicy(Enums::GSPolicyType policy)
    {
        switch(policy) {
        case Enums::GSP_RR:
            return new RRGlobalPolicy();
        case Enums::GSP_NORM:
            return new NormalGlobalPolicy();
        case Enums::GSP_LAX:
            return new LaxityGlobalPolicy();
        case Enums::GSP_RRCS:
            return new RRCSGlobalPolicy();   
        default:
           fatal("Unimplemented scheduling policy.\n");
        }
    }
};

#endif // __GPU_COMPUTE_GLOBAL_SCHEDULING_POLICY_HH__
