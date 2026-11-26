/**
 *  @file   larpandoracontent/LArCheating/CheatingEventSlicingTool.h
 *
 *  @brief  Header file for the cheating event slicing tool class.
 *
 *  $Log: $
 */
#ifndef LAR_CHEATING_EVENT_SLICING_TOOL_H
#define LAR_CHEATING_EVENT_SLICING_TOOL_H 1

#include "larpandoracontent/LArControlFlow/EventSlicingBaseTool.h"

#include <unordered_map>

namespace lar_content
{

/**
 *  @brief  CheatingEventSlicingTool class
 */
class CheatingEventSlicingTool : public EventSlicingBaseTool
{
public:
    void RunSlicing(const pandora::Algorithm *const pAlgorithm, const HitTypeToNameMap &caloHitListNames,
        const HitTypeToNameMap &clusterListNames, SliceList &sliceList);

private:
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    typedef std::unordered_map<const pandora::MCParticle *, Slice> MCParticleToSliceMap;

    /**
     *  @brief  Initialize the map from parent mc particles to slice objects
     *
     *  @param  pAlgorithm address of the calling algorithm
     *  @param  caloHitListNames the hit type to calo hit list name map
     *  @param  mcParticleToSliceMap to receive the parent mc particle to slice map
     */
    void InitializeMCParticleToSliceMap(const pandora::Algorithm *const pAlgorithm, const HitTypeToNameMap &caloHitListNames,
        MCParticleToSliceMap &mcParticleToSliceMap) const;

    /**
     *  @brief  Fill slices using hits from a specified view
     *
     *  @param  pAlgorithm address of the calling algorithm
     *  @param  hitType the hit type (i.e. view)
     *  @param  caloHitListNames the hit type to calo hit list name map
     *  @param  mcParticleToSliceMap to receive the parent mc particle to slice map
     */
    void FillSlices(const pandora::Algorithm *const pAlgorithm, const pandora::HitType hitType, const HitTypeToNameMap &caloHitListNames,
        MCParticleToSliceMap &mcParticleToSliceMap) const;

    unsigned int m_minHitsPerView{1}; ///< The minimum number of hits in a view for it to count towards keeping a slice
    bool m_requireAllViews{false};    ///< Whether every view must have enough hits to keep a slice (otherwise any view will do)
};

} // namespace lar_content

#endif // #ifndef LAR_CHEATING_EVENT_SLICING_TOOL_H
