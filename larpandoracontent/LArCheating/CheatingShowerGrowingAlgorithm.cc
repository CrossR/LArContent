/**
 *  @file   larpandoracontent/LArCheating/CheatingShowerGrowingAlgorithm.cc
 *
 *  @brief  Implementation file for the cheating shower growing algorithm.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"
#include "Helpers/MCParticleHelper.h"
#include "Objects/MCParticle.h"

#include "larpandoracontent/LArCheating/CheatingShowerGrowingAlgorithm.h"
#include "larpandoracontent/LArTrackShowerId/ShowerGrowingAlgorithm.h"
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

using namespace pandora;

namespace lar_content
{

CheatingShowerGrowingAlgorithm::CheatingShowerGrowingAlgorithm() :
    m_maxClusterFraction(0.25f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CheatingShowerGrowingAlgorithm::Run()
{
    for (const std::string &clusterListName : m_inputClusterListNames)
    {
        try
        {
            const ClusterList *pClusterList = nullptr;
            PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=, PandoraContentApi::GetList(*this, clusterListName, pClusterList));

            if (!pClusterList || pClusterList->empty())
            {
                if (PandoraContentApi::GetSettings(*this)->ShouldDisplayAlgorithmInfo())
                    std::cout << "ShowerGrowingAlgorithm: unable to find cluster list " << clusterListName << std::endl;

                continue;
            }

            // for i, cluster in enumerate(clusterList):
            //
            //     clustersToMerge = []
            //     clusterMc = getMainMCParticle(cluster)
            //
            //     for j, otherCluster in enumerate(clusterList[i:]):
            //
            //         otherClusterMC = getMainMCParticleCluster(otherCluster)
            //
            //         if clusterMC == otherClusterMC:
            //             clustersToMerge.append(j)
            //
            //     mergeClusters(cluster, clustersToMerge)
            
            // Cache Cluster -> MC.
            // Store clusters that should be merged, 
            // What to do with clusters that are never merged? 
            //     - Have a fallback check for second main MC, if over X%?

        }
        catch (StatusCodeException &statusCodeException)
        {
            throw statusCodeException;
        }
    }

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

const MCParticle* CheatingShowerGrowingAlgorithm::GetMCForCluster(const Cluster *const cluster, std::map<const Cluster*,
    const MCParticle*> &clusterToMCMap) const
{
    const MCParticle* clusterMC = nullptr;

    if (clusterToMCMap.count(cluster) > 0)
    {
        clusterMC = clusterToMCMap.at(cluster);
    }
    else
    {
        try
        {
            clusterMC = MCParticleHelper::GetMainMCParticle(cluster);
            clusterToMCMap[cluster] = clusterMC;
        }
        catch (StatusCodeException e)
        {
            std::cout << "Failed to get MC particle for cluster: " << e.ToString() << std::endl;
        }
    }

    return clusterMC;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void CheatingShowerGrowingAlgorithm::CheatedShowerGrowing(const pandora::ClusterList *const pClusterList) const
{
    std::map<const Cluster*, const MCParticle*> clusterToMCParticleMap;

    std::map<const Cluster*, bool> clusterIsUsed;
    std::map<const Cluster*, std::vector<const Cluster*>> clustersToMerge;

    for (auto it = pClusterList->begin(); it != pClusterList->end(); ++it)
    {
        const Cluster *cluster = *it;

        if (clusterIsUsed.count(cluster) > 0)
            continue;

        const MCParticle *clusterMC(this->GetMCForCluster(cluster, clusterToMCParticleMap));

        for (auto it2 = std::next(it); it2 != pClusterList->end(); ++it2) {
            const Cluster *const otherCluster = *it2;

            if (clusterIsUsed.count(otherCluster) > 0)
                continue;

            const MCParticle *otherClusterMC(this->GetMCForCluster(otherCluster, clusterToMCParticleMap));

            if (clusterMC == otherClusterMC)
            {
                clusterIsUsed[cluster] = true;
                clusterIsUsed[otherCluster] = true;
                clustersToMerge[cluster].push_back(otherCluster);
            }
        }
    }

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

// TODO: Should this be cheated? Its not used for the shower growing itself seemingly, rather instead used by
//       other algorithms. Probably just copy/fallback to the current shower growing here, is possible?
ShowerGrowingAlgorithm::AssociationType CheatingShowerGrowingAlgorithm::AreClustersAssociated(const Cluster *const /*pClusterSeed*/, const Cluster *const /*pCluster*/) const
{
    return STRONG;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode CheatingShowerGrowingAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadVectorOfValues(xmlHandle, "InputClusterListNames", m_inputClusterListNames));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MCParticleListName", m_mcParticleListName));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
