/**
 *  @file   larpandoradlcontent/LArShowerGrowing/DlShowerGrowingAlgorithm.h
 *
 *  @brief  Header file for the deep learning shower growing algorithm.
 *
 *  $Log: $
 */
#ifndef LAR_DL_SHOWER_GROWING_ALGORITHM_H
#define LAR_DL_SHOWER_GROWING_ALGORITHM_H 1

#include "Pandora/Algorithm.h"

#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"
#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

namespace lar_dl_content
{

/**
 *  @brief  DlShowerGrowingAlgorithm class
 */
class DlShowerGrowingAlgorithm : public pandora::Algorithm
{
public:
    /**
     *  @brief  Default constructor
     */
    DlShowerGrowingAlgorithm();
    virtual ~DlShowerGrowingAlgorithm();

private:
    pandora::StatusCode Run();

    /**
     *  @brief  Produce files that act as inputs to network training
     */
    pandora::StatusCode Train();

    /**
     *  @brief  Produce files that act as inputs to network training for a given view.
     *
     *  @param  clusters the cluster list
     *  @param  clusterListName the name of the current cluster
     */
    void ProduceTrainingFile(const pandora::ClusterList *clusters, const std::string &clusterListName) const;

    /**
     *  @brief  Run network inference
     */
    pandora::StatusCode Infer();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    /**
     *  @brief  If the track/shower ID is corrrect.
     */
    double IsTaggedCorrectly(const int cId, const int mcID) const;

    /**
     *  @brief  Get a unique ID for a given MC particle.
     */
    double GetIdForMC(const pandora::MCParticle *mc, std::map<const pandora::MCParticle *, int> &idMap) const;

    void Test(const pandora::ClusterList *clusters) const;

    /**
     *  @brief  Populates the MC information.
     */
    void GetMCMaps(const pandora::ClusterList *clusterList, const std::string &clusterListName,
        lar_content::LArMCParticleHelper::CaloHitToMCMap &caloToMCMap, lar_content::LArMCParticleHelper::MCContributionMap &MCtoCaloMap) const;

    /**
     *  @brief  Dump the current cluster list to a CSV, to aide visualisation or training.
     *
     *  @param  clusters the cluster list
     *  @param  clusterListName the name of the current cluster
     */
    void DumpClusterList(const pandora::ClusterList *clusters, const std::string &clusterListName) const;

    pandora::StringVector m_clusterListNames; ///< The names of the input cluster lists
    std::string m_modelFileNameU;             ///< Model file name for U view
    std::string m_modelFileNameV;             ///< Model file name for V view
    std::string m_modelFileNameW;             ///< Model file name for W view
    LArDLHelper::TorchModel m_modelU;         ///< Model for the U view
    LArDLHelper::TorchModel m_modelV;         ///< Model for the V view
    LArDLHelper::TorchModel m_modelW;         ///< Model for the W view
    bool m_visualize;                         ///< Whether to visualize the track shower ID scores
    bool m_useTrainingMode;                   ///< Training mode
    bool m_dumpClusterList = false;           ///< Dump the cluster list to a file.
    std::string m_recoStatus;                 ///< The current reconstruction status
    std::string m_trainingOutputFile;         ///< Output file name for training examples
};

} // namespace lar_dl_content

#endif // LAR_DL_SHOWER_GROWING_ALGORITHM_H
