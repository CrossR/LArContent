/**
 *  @file   larpandoracontent/LArThreading/StdThreadingManager.cc
 *
 *  @brief  Implementation of the std::thread-based threading manager class.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArThreading/StdThreadingManager.h"
#include <mutex>

namespace lar_content
{

StdThreadingManager::StdThreadingManager() :
    ThreadingManager(),
    m_shutdown(false)
{
    // Defaults to the number of hardware threads, but can be overridden.
    const unsigned int numThreads = m_maxJobCount;
    m_threads.reserve(numThreads);

    for (unsigned int i = 0; i < numThreads; ++i)
        m_threads.emplace_back(&StdThreadingManager::WorkerThread, this);
}

//------------------------------------------------------------------------------------------------------------------------------------------

StdThreadingManager::~StdThreadingManager()
{
    this->WaitForCompletion();

    {
        std::unique_lock<std::mutex> lock(m_jobMutex);
        m_shutdown = true;
    }

    m_jobCondition.notify_all();

    for (auto &thread : m_threads)
    {
        if (thread.joinable())
            thread.join();
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void StdThreadingManager::SubmitJobImpl(std::function<void()> job)
{
    {
        std::unique_lock<std::mutex> lock(m_jobMutex);
        m_jobs.push_back(std::move(job));
    }

    m_jobCondition.notify_one();
}

//------------------------------------------------------------------------------------------------------------------------------------------

void StdThreadingManager::WaitForCompletion()
{
    std::unique_lock<std::mutex> lock(m_jobMutex);

    m_completionCondition.wait(lock, [this] { return (m_jobs.empty() && m_runningJobCount.load() == 0); });
}

//------------------------------------------------------------------------------------------------------------------------------------------

void StdThreadingManager::NotifyJobCompletion()
{
    std::unique_lock<std::mutex> lock(m_jobMutex);
    if (m_jobs.empty() && m_runningJobCount.load() == 0)
        m_completionCondition.notify_all();
}

//------------------------------------------------------------------------------------------------------------------------------------------

void StdThreadingManager::WorkerThread()
{
    while (true)
    {
        std::function<void()> job;

        {
            std::unique_lock<std::mutex> lock(m_jobMutex);
            m_jobCondition.wait(lock, [this] { return m_shutdown || !m_jobs.empty(); });

            if (m_shutdown && m_jobs.empty())
                return;

            job = std::move(m_jobs.front());
            m_jobs.pop_front();
        }

        if (job)
        {
            job();

            {
                std::unique_lock<std::mutex> lock(m_jobMutex);
                if (m_jobs.empty() && m_runningJobCount.load() == 0)
                    m_completionCondition.notify_all();
            }
        }
    }
}

} // namespace lar_content