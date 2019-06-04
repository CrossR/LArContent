/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/ConsolidatedHitsBaseTool.cc
 *
 *  @brief  Implementation of the consolidated hits base tool.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/ConsolidatedHitsBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

using namespace pandora;

namespace lar_content
{

ConsolidatedHitsBaseTool::ConsolidatedHitsBaseTool() {}

//------------------------------------------------------------------------------------------------------------------------------------------

void ConsolidatedHitsBaseTool::GetTrackHits3D(
        const CaloHitVector &inputTwoDHits,
        const MatchedSlidingFitMap &inputSlidingFitMap,
        ProtoHitVector &protoHitVector
) const
{
    std::cout << "*****************************************************************************************************************************" << std::endl;
    std::cout << "ConsolidatedHitsBaseTool is being ran!" << std::endl;
    std::cout << "*****************************************************************************************************************************" << std::endl;

    if (inputTwoDHits.size() < 0) std::cout << "inputTwoDHits was empty!" << std::endl;
    if (inputSlidingFitMap.size() < 0) std::cout << "inputTwoDHits was empty!" << std::endl;
    if (protoHitVector.size() < 0) std::cout << "inputTwoDHits was empty!" << std::endl;

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ConsolidatedHitsBaseTool::ReadSettings(const TiXmlHandle xmlHandle)
{
    return TrackHitsBaseTool::ReadSettings(xmlHandle);
}

} // namespace lar_content
