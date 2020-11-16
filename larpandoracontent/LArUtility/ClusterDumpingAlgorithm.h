/**
 *  @file   larpandoracontent/LArUtility/ClusterDumpingAlgorithm.h
 *
 *  @brief  Header file for the debug cluster dumping algorithm class.
 *          TODO: Remove this, implementation and entries in LArContent.
 *
 *  $Log: $
 */
#ifndef LAR_CLUSTER_DUMPING_ALGORITHM_H
#define LAR_CLUSTER_DUMPING_ALGORITHM_H 1

#include "Pandora/Algorithm.h"

#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"

namespace lar_content
{

/**
 *  @brief  ClusterDumpingAlgorithm::Algorithm class
 */
class ClusterDumpingAlgorithm : public pandora::Algorithm
{
private:
    pandora::StatusCode Run();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    /**
     *  @brief  Dump the current cluster list to a CSV, to aide visualisation or training.
     *
     *  @param  clusterListName the name of the current cluster
     */
    void DumpClusterList(const std::string &clusterListName) const;

    /**
     *  @brief  If the track/shower ID is corrrect.
     *
     */
    double IsTaggedCorrectly(const int cId, const int mcID) const;

    /**
     *  @brief  Populates the MC information.
     *
     */
    void GetMCMaps(const pandora::ClusterList *clusterList,
        LArMCParticleHelper::CaloHitToMCMap &caloToMCMap, LArMCParticleHelper::MCContributionMap &MCtoCaloMap) const;

    std::string           m_recoStatus;        ///< The current reconstruction status
    pandora::StringVector m_clusterListNames;  ///< The names of the input cluster lists
};

} // namespace lar_content

#endif // #ifndef LAR_CLUSTER_DUMPING_ALGORITHM_H
