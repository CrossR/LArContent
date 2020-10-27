/**
 *  @file   larpandoracontent/LArCheating/CheatingShowerGrowingAlgorithm.h
 *
 *  @brief  Header file for the cheating shower growing algorithm.
 *
 *  $Log: $
 */
#ifndef LAR_CHEATING_SHOWER_GROWING_ALGORITHM_H
#define LAR_CHEATING_SHOWER_GROWING_ALGORITHM_H 1

#include "larpandoracontent/LArControlFlow/MasterAlgorithm.h"

#include "larpandoracontent/LArTrackShowerId/BranchGrowingAlgorithm.h"

namespace lar_content
{

/**
 *  @brief  CheatingShowerGrowingAlgorithm class
 */
class CheatingShowerGrowingAlgorithm : public BranchGrowingAlgorithm
{
public:
    /**
     *  @brief  Default constructor
     */
    CheatingShowerGrowingAlgorithm();
private:
    pandora::StatusCode Run();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    AssociationType AreClustersAssociated(const pandora::Cluster *const pClusterSeed, const pandora::Cluster *const pCluster) const;

    pandora::StringVector  m_inputClusterListNames; ///< The names of the input cluster lists.
    std::string            m_mcParticleListName;    ///< Input MC particle list name.
    float                  m_maxClusterFraction;    ///< The maximum fraction a cluster can be contaminated by to be considered clean.
};

} // namespace lar_content

#endif // #ifndef LAR_CHEATING_SHOWER_GROWING_ALGORITHM_H
