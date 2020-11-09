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

namespace lar_content
{

/**
 *  @brief  CheatingShowerGrowingAlgorithm class
 */
class CheatingShowerGrowingAlgorithm : public pandora::Algorithm
{
public:
    /**
     *  @brief  Default constructor
     */
    CheatingShowerGrowingAlgorithm();
private:
    pandora::StatusCode Run();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    /**
     *  @brief  Get the MC particle for a given cluster, caching to a map.
     *
     *  @param  cluster         The current cluster to lookup
     *  @param  clusterToMCMap  Map from Cluster to MC to cache results.
     */
    const pandora::MCParticle* GetMCForCluster(const pandora::Cluster *const cluster, std::map<const pandora::Cluster*,
        const pandora::MCParticle*> &clusterToMCMap) const;

    /**
     *  @brief  Cheated shower growing. Use MC to match clusters based on the main MC particle.
     *
     *  @param  pClusterList the list of clusters
     */
    void CheatedShowerGrowing(const pandora::ClusterList *const pClusterList, const std::string &listName) const;

    pandora::StringVector  m_inputClusterListNames; ///< The names of the input cluster lists.
    std::string            m_mcParticleListName;    ///< Input MC particle list name.
    float                  m_maxClusterFraction;    ///< The maximum fraction a cluster can be contaminated by to be considered clean.
};

} // namespace lar_content

#endif // #ifndef LAR_CHEATING_SHOWER_GROWING_ALGORITHM_H
