/**
 *  @file   larpandoracontent/LArThreading/TbbThreadingManager.h
 *
 *  @brief  Header file for the Intel TBB-based threading manager class.
 *
 *  $Log: $
 */
#ifndef LAR_TBB_THREADING_MANAGER_H
#define LAR_TBB_THREADING_MANAGER_H 1

#include "larpandoracontent/LArThreading/ThreadingManager.h"

#ifdef PANDORA_USE_TBB
#include <tbb/task_group.h>
#endif

namespace lar_content
{

/**
 *  @brief  TbbThreadingManager class - implementation using Intel TBB
 */
class TbbThreadingManager : public ThreadingManager
{
public:
    /**
     *  @brief  Default constructor
     */
    TbbThreadingManager();

    /**
     *  @brief  Destructor
     */
    ~TbbThreadingManager();

    /**
     *  @brief  Wait for all submitted jobs to complete
     */
    void WaitForCompletion() override;

protected:
    /**
     *  @brief  Submit job for execution
     *
     *  @param  job the job to submit as a callable
     */
    void SubmitJobImpl(std::function<void()> job) override;

private:
#ifdef PANDORA_USE_TBB
    tbb::task_group m_taskGroup;    ///< TBB task group
#endif
};

} // namespace lar_content

#endif // #ifndef LAR_TBB_THREADING_MANAGER_H