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

    pandora::StatusCode GetGraphData(const pandora::CaloHitList &caloHits, std::vector<pandora::CartesianVector> &pos,
        std::vector<std::array<float, 1>> &node_features, std::vector<std::pair<int, int>> &edges);

    pandora::StatusCode BuildGraph(LArDLHelper::TorchInputVector &inputs,
        std::vector<pandora::CartesianVector> &pos, std::vector<std::array<float, 1>> &node_features,
        std::vector<std::pair<int, int>> &edges);

    LArDLHelper::TorchModel m_modelFile;  ///< The model to use.

    const float m_scalingFactor;          ///< The scaling factor for the input.
    const std::string m_caloHitListName; ///< The name of the input CaloHit list.
};

} // namespace lar_dl_content

#endif // LAR_DL_SLICING_ALGORITHM_H
