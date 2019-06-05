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

#include "larpandoracontent/LArThreeDReco/LArHitCreation/ClearLongitudinalTrackHitsTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ClearTransverseTrackHitsTool.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/MultiValuedLongitudinalTrackHitsTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/MultiValuedTransverseTrackHitsTool.h"

using namespace pandora;

namespace lar_content
{

ConsolidatedHitsBaseTool::ConsolidatedHitsBaseTool() {}

//------------------------------------------------------------------------------------------------------------------------------------------
void ConsolidatedHitsBaseTool::GetTrackHits3D(
        const CaloHitVector &inputTwoDHits,
        const MatchedSlidingFitMap &inputSlidingFitMap,
        ProtoHitVector &protoHitVector
        // protoHitVectorMap
        // protoHitVector to be populated
) const
{
    std::cout << "ConsolidatedHitsBaseTool is being ran!" << std::endl;

    // For now, lets test out running the default set of algorithms.
    // This should give the exact same results as the ordering defined
    // in the XML file.
    auto clearLong = ClearLongitudinalTrackHitsTool();
    auto clearTransverse = ClearTransverseTrackHitsTool();
    auto multiValueLong = MultiValuedLongitudinalTrackHitsTool();
    auto multiValueTransverse = MultiValuedTransverseTrackHitsTool();

    clearLong.m_minViews = 3;
    clearTransverse.m_minViews = 3;
    multiValueLong.m_minViews = 3;
    multiValueTransverse.m_minViews = 3;

    clearTransverse.GetTrackHits3D(inputTwoDHits, inputSlidingFitMap, protoHitVector);
    clearLong.GetTrackHits3D(inputTwoDHits, inputSlidingFitMap, protoHitVector);
    multiValueLong.GetTrackHits3D(inputTwoDHits, inputSlidingFitMap, protoHitVector);
    multiValueTransverse.GetTrackHits3D(inputTwoDHits, inputSlidingFitMap, protoHitVector);

    clearLong.m_minViews = 2;
    clearTransverse.m_minViews = 2;
    multiValueLong.m_minViews = 2;

    clearTransverse.GetTrackHits3D(inputTwoDHits, inputSlidingFitMap, protoHitVector);
    clearLong.GetTrackHits3D(inputTwoDHits, inputSlidingFitMap, protoHitVector);
    multiValueLong.GetTrackHits3D(inputTwoDHits, inputSlidingFitMap, protoHitVector);

    return;
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ConsolidatedHitsBaseTool::ReadSettings(const TiXmlHandle xmlHandle)
{
    return TrackHitsBaseTool::ReadSettings(xmlHandle);
}

} // namespace lar_content
