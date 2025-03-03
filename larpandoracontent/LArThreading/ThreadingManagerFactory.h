/**
 *  @file   larpandoracontent/LArThreading/ThreadingManagerFactory.h
 *
 *  @brief  Header file for the threading manager factory.
 *
 *  $Log: $
 */
#ifndef LAR_THREADING_MANAGER_FACTORY_H
#define LAR_THREADING_MANAGER_FACTORY_H 1

#include "larpandoracontent/LArThreading/ThreadingManager.h"
#include "larpandoracontent/LArThreading/StdThreadingManager.h"
#ifdef PANDORA_USE_TBB
#include "larpandoracontent/LArThreading/TbbThreadingManager.h"
#endif

namespace lar_content
{

/**
 *  @brief  ThreadingManagerFactory class
 */
class ThreadingManagerFactory
{
public:
    /**
     *  @brief  Create a threading manager
     *
     *  @return shared pointer to a threading manager
     */
    static std::shared_ptr<ThreadingManager> CreateThreadingManager()
    {
#ifdef PANDORA_USE_TBB
        return std::make_shared<TbbThreadingManager>();
#else
        return std::make_shared<StdThreadingManager>();
#endif
    }
};

} // namespace lar_content

#endif // #ifndef LAR_THREADING_MANAGER_FACTORY_H