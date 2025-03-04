/**
 *  @file   larpandoracontent/LArThreading/TbbThreadingManager.cc
 *
 *  @brief  Implementation of the Intel TBB-based threading manager class.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArThreading/TbbThreadingManager.h"

#include <cassert>

namespace lar_content
{

TbbThreadingManager::TbbThreadingManager() :
    ThreadingManager()
{
#ifdef PANDORA_USE_TBB
    // Update the number of threads to match the maximum job count,
    // so we will continue to submit jobs even if some are waiting.
    //
    // In regular C++ threading, we would have to wait for a job to complete
    // before submitting a new one, but TBB can handle this for us.
    m_maxJobCount = tbb::task_scheduler_init::default_num_threads();
#endif
}

//------------------------------------------------------------------------------------------------------------------------------------------

TbbThreadingManager::~TbbThreadingManager()
{
    this->WaitForCompletion();
}

//------------------------------------------------------------------------------------------------------------------------------------------

void TbbThreadingManager::SubmitJobImpl(std::function<void()> job)
{
#ifdef PANDORA_USE_TBB
    m_taskGroup.run(std::move(job));
#else
    // Fall back to immediate execution if TBB not available
    job();
#endif
}

//------------------------------------------------------------------------------------------------------------------------------------------

void TbbThreadingManager::WaitForCompletion()
{
#ifdef PANDORA_USE_TBB
    m_taskGroup.wait();
#endif

    assert(m_runningJobCount.load() == 0);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void TbbThreadingManager::NotifyJobCompletion()
{
    // INFO: No-op, as TBB handles this for us.
}

} // namespace lar_content