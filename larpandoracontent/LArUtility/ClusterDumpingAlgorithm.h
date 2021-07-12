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
     *  @param  clusters the cluster list
     *  @param  clusterListName the name of the current cluster
     */
    void DumpClusterList(const pandora::ClusterList *clusters, const std::string &clusterListName) const;

    /**
     *  @brief  Just produce the CSV files for training.
     *
     *  @param  clusters the cluster list
     *  @param  clusterListName the name of the current cluster
     */
    void ProduceTrainingFile(const pandora::ClusterList *clusters, const std::string &clusterListName) const;

    /**
     *  @brief  If the track/shower ID is corrrect.
     */
    double IsTaggedCorrectly(const int cId, const int mcID) const;

    /**
     *  @brief  Get a unique ID for a given MC particle.
     */
    double GetIdForMC(const pandora::MCParticle* mc, std::map<const pandora::MCParticle*, int> &idMap) const;

    void Test(const pandora::ClusterList *clusters) const;

    /**
     *  @brief  Populates the MC information.
     *
     */
    void GetMCMaps(const pandora::ClusterList *clusterList, const std::string &clusterListName,
        LArMCParticleHelper::CaloHitToMCMap &caloToMCMap, LArMCParticleHelper::MCContributionMap &MCtoCaloMap) const;

    bool                  m_dumpClusterList = false; ///< Dump the cluster list to a file.
    std::string           m_trainFileName = "";      ///< Name of training file, if set will only produce training files.
    std::string           m_recoStatus;              ///< The current reconstruction status
    pandora::StringVector m_clusterListNames;        ///< The names of the input cluster lists
};

} // namespace lar_content

#endif // #ifndef LAR_CLUSTER_DUMPING_ALGORITHM_H
