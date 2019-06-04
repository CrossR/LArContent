/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/ConsolidatedHitsBaseTool.h
 *
 *  @brief  Header file for the consolidated hits base tool.
 *
 *  $Log: $
 */
#ifndef CONSOLIDATED_HITS_BASE_TOOL_H
#define CONSOLIDATED_HITS_BASE_TOOL_H 1

#include "larpandoracontent/LArThreeDReco/LArHitCreation/TrackHitsBaseTool.h"

namespace lar_content
{

class ConsolidatedHitsBaseTool : public TrackHitsBaseTool
{
public:

    ConsolidatedHitsBaseTool();

protected:

    virtual void GetTrackHits3D(
            const pandora::CaloHitVector &inputTwoDHits,
            const MatchedSlidingFitMap &matchedSlidingFitMap,
            ProtoHitVector &protoHitVector
    ) const;

    virtual pandora::StatusCode ReadSettings(
            const pandora::TiXmlHandle xmlHandle
    );
};

} // namespace lar_content

#endif // #ifndef CONSOLIDATED_HITS_BASE_TOOL_H
