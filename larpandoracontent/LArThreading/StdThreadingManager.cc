/**
 *  @file   larpandoracontent/LArThreading/StdThreadingManager.cc
 *
 *  @brief  Implementation of the std::thread-based threading manager class.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArThreading/StdThreadingManager.h"

namespace lar_content
{

StdThreadingManager::StdThreadingManager() :
    ThreadingManager(),
    m_shutdown(false)
{
    const unsigned int numThreads = std::thread::hardware_concurrency();
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
    while (true)
    {
        bool jobsRemaining = false;

        {
            std::unique_lock<std::mutex> lock(m_jobMutex);
            jobsRemaining = !m_jobs.empty() || (m_runningJobCount.load() > 0);
        }

        if (!jobsRemaining)
            break;

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
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
            job();
    }
}

} // namespace lar_content