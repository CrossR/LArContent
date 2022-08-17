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
        const pandora::Cluster *cluster;
        int clusterNum;
        Eigen::MatrixXf hits;
        pandora::CartesianVector direction;
        float numOfHits;
        float orientation;
        float xMean;
        float zMean;
        float vertexDisplacement;
    };

    /** @struct MatrixIndex
     *  @brief Simple struct to be used in an Eigen visitor, to evaluate a matrix.
     */
    struct MatrixIndex
    {
        int row;
        int col;
    };

    typedef std::map<int, const pandora::Cluster *> IdClusterMap;
    typedef std::vector<NodeFeature> NodeFeatureVector;
    typedef std::vector<std::vector<int>> EdgeVector;
    typedef std::vector<std::vector<float>> EdgeFeatureVector;

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
     *  @brief  Generate the general graph data for the given input cluster list.
     *          The graph itself is made elsewhere, to allow a decision about the input cluster to be made.
     *
     *  @param  clusters the cluster list
     *  @param  vertex the current vertex to use
     *  @param  nodeToCluster a map for each node ID to its parent cluster, to allow merging later
     *  @param  nodes vector of node features to populate
     *  @param  edges vector of node indicies to build edges between
     *  @param  edgeFeatures vector of edge features to populate
     */
    void GetGraphData(const pandora::ClusterList &clusters, const pandora::Vertex *vertex, IdClusterMap &nodeToCluster,
        NodeFeatureVector &nodes, EdgeVector &edges, EdgeFeatureVector &edgeFeatures);

    /**
     *  @brief  Select an input cluster from the complete cluster list. This should be the best cluster to grow
     *          from the input list, which is defined as a combination between shower-like, size and shape.
     *          Also count up the total hits that are going to be considered this iteration, for later use.
     *
     *  @param  clusters the cluster list
     *  @param  inputClusterNum to be set to the selected input cluster
     *  @param  totalHits a count of all the hits in every considered cluster
     */
    void GetInputCluster(const pandora::ClusterList &clusters, int &inputClusterNum, int &totalHits);

    /**
     *  @brief  Build a graph for the given input cluster.
     *
     *  @param  inputClusterNum the input cluster id, which will be grown
     *  @param  nodes vector of node features to use
     *  @param  edges vector of node indicies to use
     *  @param  edgeFeatures vector of edge features to use
     *  @param  inputs the final torch input vector to infer from
     */
    void BuildGraph(const int inputClusterNum, NodeFeatureVector &nodes, EdgeVector &edges, EdgeFeatureVector &edgeFeatures,
            LArDLHelper::TorchInputVector &inputs);

    /**
     *  @brief  Grow clusters using the network output.
     *
     *  @param  listName the name of the cluster list to grow in
     *  @param  inputCluster the input cluster, to grow
     *  @param  nodeMap map of cluster ID to cluster pointer
     *  @param  output the output of the network
     *  @param  clusters the cluster list, now to be updated.
     */
    pandora::StatusCode GrowClusters(const std::string &listName, const pandora::Cluster *inputCluster, IdClusterMap &nodeMap,
        LArDLHelper::TorchOutput &output, pandora::ClusterList &clusters);

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
    bool m_limitedEdges;                      ///< Use limited internal edges
    std::string m_recoStatus;                 ///< The current reconstruction status
    std::string m_trainingOutputFile;         ///< Output file name for training examples
    std::string m_dumpScoreFile;              ///< TODO: Remove. File path to store score files in.

    // INFO: Distances here are relative to the scaled position, not to their original value.

    const int m_rounding = 2;                                                  ///< How much to round coordinates. (i.e. round to closest 2/5/10 etc.)
    const int m_kNN = 9;                                                       ///< What K is, for the K Nearest Neighbour
    const float m_distanceCutOff = std::numeric_limits<float>::max();          ///< Maximum distance between nodes to allow an edge
    const float m_vertexProtectionRadius = std::numeric_limits<float>::max();  ///< Distance to be considered "close" to the vertex
    const float m_vertexProtectionAngle = 0.3;                                 ///< Angle in radians to be considered "steep"
                                                                               ///< Don't allow clusters that aren't aligned near the vertex

    const float m_scalingFactor = 500.f;                                       ///< Scale coordinates by this factor (stop exploding values)
};

} // namespace lar_dl_content

#endif // LAR_DL_SHOWER_GROWING_ALGORITHM_H
