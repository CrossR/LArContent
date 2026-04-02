/**
 *  @file   larpandoradlcontent/LArSlicing/DLSlicingAlgorithm.h
 *
 *  @brief  Header file for the deep learning slicing algorithm.
 *
 *  $Log: $
 */
#ifndef LAR_DL_SLICING_ALGORITHM_H
#define LAR_DL_SLICING_ALGORITHM_H 1

#include "Pandora/Algorithm.h"
#include "Pandora/AlgorithmHeaders.h"

#include "larpandoradlcontent/LArHelpers/LArDLHelper.h"

using namespace lar_content;

namespace lar_dl_content
{
/**
 *  @brief  DeepLearningSLicingAlgorithm class
 */
class DlSlicingAlgorithm : public pandora::Algorithm
{
public:
    /**
     *  @brief Default constructor
     */
    DlSlicingAlgorithm();

private:
    pandora::StatusCode Run();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);
    pandora::StatusCode Infer();

    /**
     *  @brief Build the graph data (nodes, edges, and features) from the input calo hits.
     *
     *  @param caloHits The input calo hits.
     *  @param pos The positions of the nodes.
     *  @param node_features The features for each node.
     *  @param edges The edges to be built.
     */
    pandora::StatusCode GetGraphData(const pandora::CaloHitList &caloHits, std::vector<pandora::CartesianVector> &pos,
        std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges);

    /**
     *  @brief Process the given graph data into the expected tensor format for the model, and insert into the input vector.
     *
     *  @param inputs The input vector to be filled with the graph data.
     *  @param pos The positions of the nodes.
     *  @param node_features The features for each node.
     *  @param edges The edges between nodes.
     */
    pandora::StatusCode BuildGraph(LArDLHelper::TorchInputVector &inputs, std::vector<pandora::CartesianVector> &pos,
        std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges);

    LArDLHelper::TorchModel m_modelFile; ///< The model to use.

    float m_scalingFactor;               ///< The scaling factor for the input.
    std::vector<float> m_thresholds;     ///< Distance Class Thresholds.
    int m_nDistanceClasses;              ///< The number of distance classes (derived from thresholds).
    std::string m_caloHitListName;       ///< The name of the input CaloHit list.
    std::string m_outputClusterListName; ///< The name of the output Cluster list to create with the predicted instances.
    int m_k;                             ///< The number of nearest neighbours to use when building the graph.
};

} // namespace lar_dl_content

#endif // LAR_DL_SLICING_ALGORITHM_H
