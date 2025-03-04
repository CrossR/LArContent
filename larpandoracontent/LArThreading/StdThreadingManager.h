/**
 *  @file   larpandoracontent/LArThreading/StdThreadingManager.h
 *
 *  @brief  Header file for the std::thread-based threading manager class.
 *
 *  $Log: $
 */
#ifndef LAR_STD_THREADING_MANAGER_H
#define LAR_STD_THREADING_MANAGER_H 1

#include "larpandoracontent/LArThreading/ThreadingManager.h"

#include <condition_variable>
#include <deque>
#include <thread>
#include <vector>

namespace lar_content
{

/**
 *  @brief  StdThreadingManager class - implementation using std::thread
 */
class StdThreadingManager : public ThreadingManager
{
public:
    /**
     *  @brief  Default constructor
     */
    StdThreadingManager();

    /**
     *  @brief  Destructor
     */
    ~StdThreadingManager();

    /**
     *  @brief  Wait for all submitted jobs to complete
     */
    void WaitForCompletion() override;

    /**
     *  @brief Notify that a job has completed
     */
    void NotifyJobCompletion() override;

protected:
    /**
     *  @brief  Submit job for execution
     *
     *  @param  job the job to submit as a callable
     */
    void SubmitJobImpl(std::function<void()> job) override;

private:
    /**
     *  @brief  Worker thread function
     */
    void WorkerThread();

    std::vector<std::thread>     m_threads;              ///< Worker threads
    std::deque<std::function<void()>> m_jobs;            ///< Job queue
    std::mutex                   m_jobMutex;             ///< Mutex protecting job queue
    std::condition_variable      m_jobCondition;         ///< Condition variable for job queue
    std::condition_variable      m_completionCondition;  ///< Condition variable for job completion
    bool                         m_shutdown;             ///< Shutdown flag
};

} // namespace lar_content

#endif // #ifndef LAR_STD_THREADING_MANAGER_H