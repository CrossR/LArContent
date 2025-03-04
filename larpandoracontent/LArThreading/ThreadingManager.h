/**
 *  @file   larpandoracontent/LArThreading/ThreadingManager.h
 *
 *  @brief  Header file for the threading manager class.
 *
 *  $Log: $
 */
#ifndef LAR_THREADING_MANAGER_H
#define LAR_THREADING_MANAGER_H 1

#include "Pandora/StatusCodes.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

namespace lar_content
{

/**
 *  @brief  ThreadingManager class
 */
class ThreadingManager
{
public:
    /**
     *  @brief  Default constructor
     */
    ThreadingManager();

    /**
     *  @brief  Destructor
     */
    virtual ~ThreadingManager() = default;

    /**
     *  @brief  Submit a job to be executed asynchronously
     *
     *  @param  function the function to execute
     *  @param  args the arguments to pass to the function
     */
    template <typename Function, typename... Args>
    void SubmitJob(Function &&function, Args &&...args);

    /**
     *  @brief  Wait for all submitted jobs to complete
     */
    virtual void WaitForCompletion() = 0;

    /**
     *  @brief  Notify that a job has completed
     */
    virtual void NotifyJobCompletion() = 0;

    /**
     *  @brief  Get the number of currently running jobs
     *
     *  @return the number of running jobs
     */
    unsigned int GetRunningJobCount() const;

    /**
     *  @brief  Get the maximum number of concurrent jobs allowed
     *
     *  @return the maximum number of jobs
     */
    unsigned int GetMaxJobCount() const;

    /**
     *  @brief  Set the maximum number of concurrent jobs allowed
     *
     *  @param  maxJobCount the maximum number of jobs
     */
    void SetMaxJobCount(unsigned int maxJobCount);

protected:
    /**
     *  @brief  Implementation-specific job submission
     *
     *  @param  job the job to submit as a callable
     */
    virtual void SubmitJobImpl(std::function<void()> job) = 0;

    std::atomic<unsigned int> m_runningJobCount; ///< Number of currently running jobs
    unsigned int m_maxJobCount;                  ///< Maximum number of concurrent jobs
    std::condition_variable m_jobSlotCondition;  ///< Condition variable for job slot availability
    std::mutex m_mutex;                          ///< Mutex for thread safety
};

//------------------------------------------------------------------------------------------------------------------------------------------

inline ThreadingManager::ThreadingManager() :
    m_runningJobCount(0),
    m_maxJobCount(std::thread::hardware_concurrency())
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

template <typename Function, typename... Args>
inline void ThreadingManager::SubmitJob(Function &&function, Args &&...args)
{
    std::unique_lock<std::mutex> submissionLock(m_mutex);

    // Wait until we have room to add another job
    // TODO: Tune this? Could be useful to bail on really bad cases though...
    auto timeout = std::chrono::seconds(120);
    m_jobSlotCondition.wait_for(submissionLock, timeout, [this]() { return m_runningJobCount < m_maxJobCount; });

    // Create a bound function object
    auto boundFunction = std::bind(std::forward<Function>(function), std::forward<Args>(args)...);

    // Prepare the job with its arguments
    std::function<void()> job = [this, boundFunction]()
    {
        m_runningJobCount++;

        try
        {
            boundFunction();
        }
        catch (const pandora::StatusCodeException &statusCodeException)
        {
            std::cerr << "ThreadingManager: Exception from job " << statusCodeException.ToString() << std::endl;
        }
        m_runningJobCount--;

        {
            // Notify that a job slot could now be available
            std::lock_guard<std::mutex> completionLock(m_mutex);
            m_jobSlotCondition.notify_one();
        }

        // Notify for anything waiting that a job has completed
        this->NotifyJobCompletion();
    };

    // Submit the job
    this->SubmitJobImpl(std::move(job));
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline unsigned int ThreadingManager::GetRunningJobCount() const
{
    return m_runningJobCount.load();
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline unsigned int ThreadingManager::GetMaxJobCount() const
{
    return m_maxJobCount;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline void ThreadingManager::SetMaxJobCount(unsigned int maxJobCount)
{
    m_maxJobCount = maxJobCount;
}

} // namespace lar_content

#endif // #ifndef LAR_THREADING_MANAGER_H