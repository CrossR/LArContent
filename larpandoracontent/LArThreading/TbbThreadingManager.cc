/**
 *  @file   larpandoracontent/LArThreading/TbbThreadingManager.cc
 *
 *  @brief  Implementation of the Intel TBB-based threading manager class.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArThreading/TbbThreadingManager.h"

namespace lar_content
{

TbbThreadingManager::TbbThreadingManager() :
    ThreadingManager()
{
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

    // Double-check that all jobs completed reporting properly
    while (m_runningJobCount.load() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

} // namespace lar_content