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

#include <Eigen/Dense>

namespace lar_dl_content
{

/** @struct RoundedClusterInfo
 *  @brief This structure is used to store information about a rounded cluster.
 *         A rounded cluster in this context is where hits of a given cluster are rounded to some integer value.
 *         We then group up all the hits that now lie together after rounding.
 */
struct RoundedClusterInfo
{
    Eigen::MatrixXf hits;
    int numOfHits;
    float totalX;
    float totalZ;
};

/** @struct NodeFeature
 *  @brief This structure stores all the features of a given node of the graph.
 *         Most are direct features of the graph, but the hits field is used to
 *         calculate various distances later.
 */
struct NodeFeature
{
    int clusterId;
    Eigen::MatrixXf hits;
    pandora::CartesianVector direction;
    float numOfHits;
    float orientation;
    float xMean;
    float zMean;
    float vertexDisplacement;
};

struct MatrixIndex
{
    int row;
    int col;
};

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
     *  @brief  Run network inference for all three views.
     */
    pandora::StatusCode Infer();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    /**
     *  @brief  Run network inference for a given view.
     *
     *  @param  clusters the cluster list
     *  @param  listName the cluster list name
     */
    pandora::StatusCode InferForView(const pandora::ClusterList *clusters, const std::string &listName);

    /**
     *  @brief  Produce files that act as inputs to network training for a given view.
     *
     *  @param  clusters the cluster list
     *  @param  clusterListName the name of the current cluster
     */
    void ProduceTrainingFile(const pandora::ClusterList *clusters, const std::string &clusterListName) const;

    /**
     *  @brief  Visualise the given feature vector.
     *
     *  @param  nodeTensor the feature tensor to be visualised.
     *  @param  edgeTensor the edge tensor to be visualised.
     *  @param  output the result of the network feature vector.
     *  @param  clusterListName the name of the current cluster
     */
    void Visualize(const LArDLHelper::TorchInput nodeTensor, const LArDLHelper::TorchInput edgeTensor,
        const LArDLHelper::TorchOutput output, const std::string &clusterListName) const;

    /**
     *  @brief  If the track/shower ID is corrrect.
     *
     *  @param  cID The cluster ID
     *  @param  idMap The MC particle ID
     */
    double IsTaggedCorrectly(const int cId, const int mcID) const;

    /**
     *  @brief  Get a unique ID for a given MC particle.
     *
     *  @param  mc the mc particle pointer
     *  @param  idMap mc particle to ID map
     */
    double GetIdForMC(const pandora::MCParticle *mc, std::map<const pandora::MCParticle *, int> &idMap) const;

    /**
     *  @brief  Populates the MC information to be used later.
     *
     *  @param  clusterList the cluster list
     *  @param  clusterListName the name of the current cluster
     *  @param  caloToMCMap A calo hit to MC map, to be populated
     *  @param  MCtoCaloMap A MC to calo map, to be populated
     */
    void GetMCMaps(const pandora::ClusterList *clusterList, const std::string &clusterListName,
        lar_content::LArMCParticleHelper::CaloHitToMCMap &caloToMCMap, lar_content::LArMCParticleHelper::MCContributionMap &MCtoCaloMap) const;

    /**
     *  @brief  Dump the current cluster list to both a ROOT and CSV file, to aide analysis outside of Pandora.
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
