/**
 *  @file   larpandoracontent/LArUtility/ClusterDumpingAlgorithm.h
 *
 *  @brief  Header file for the debug cluster dumping algorithm class.
 *          TODO: Remove this, implementation and entries in LArContent.
 *
 *  $Log: $
 */
#include "Pandora/PandoraInternal.h"
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
     *  @brief  Dump information about the current cluster list.
     *          This can include CSVs and ROOT metrics (MC and Non-MC based).
     *
     *  @param  clusters the cluster list
     *  @param  clusterListName the name of the current cluster
     */
    void DumpClusterInfo(const pandora::ClusterList *clusters, const std::string &clusterListName) const;
    void DumpRecoInfo(const pandora::ClusterList *clusters, const std::string &fileName) const;

    /**
     *  @brief  If the track/shower ID is corrrect.
     */
    double IsTaggedCorrectly(const int cId, const int mcID) const;

    /**
     *  @brief  Get a unique ID for a given MC particle.
     */
    double GetIdForMC(const pandora::MCParticle *mc, std::map<const pandora::MCParticle *, int> &idMap) const;

    /**
     *  @brief  Populates the MC information.
     *
     */
    void GetMCMaps(const pandora::ClusterList *clusterList, const std::string &clusterListName,
        LArMCParticleHelper::CaloHitToMCMap &caloToMCMap, LArMCParticleHelper::MCContributionMap &MCtoCaloMap,
        const pandora::MCParticleList *mcParticleList) const;

    bool m_dumpClusterList = false;    ///< Dump the cluster list to a file.
    bool m_nonSplit = false;           ///< Is the cluster list split?
    std::string m_recoStatus;          ///< The current reconstruction status
    pandora::StringVector m_viewNames; ///< The names of the views to use
};

} // namespace lar_content

#endif // #ifndef LAR_CLUSTER_DUMPING_ALGORITHM_H
