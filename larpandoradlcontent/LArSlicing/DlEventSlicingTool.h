/**
 *  @file   larpandoradlcontent/LArSlicing/DlEventSlicingTool.h
 *
 *  @brief  Header file for the DL event slicing tool class.
 *
 *  $Log: $
 */
#ifndef LAR_DL_EVENT_SLICING_TOOL_H
#define LAR_DL_EVENT_SLICING_TOOL_H 1

#include "larpandoracontent/LArControlFlow/EventSlicingBaseTool.h"

using namespace lar_content;

namespace lar_dl_content
{

/**
 *  @brief  DLEventSlicingTool class
 */
class DLEventSlicingTool : public EventSlicingBaseTool
{
public:
    /**
     *  @brief  Default constructor
     */
    DLEventSlicingTool();

    /**
     *  @brief  Run the event slicing
     *
     *  @param  pAlgorithm the address of the calling algorithm
     *  @param  caloHitListNames the hit type to calo hit list name map
     *  @param  clusterListNames the hit type to cluster list name map
     *  @param  sliceList the slice list to receive the new slices
     */
    void RunSlicing(const pandora::Algorithm *const pAlgorithm, const HitTypeToNameMap &caloHitListNames,
        const HitTypeToNameMap &clusterListNames, SliceList &sliceList);

protected:
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    /**
     *  @brief  Called once per event before any hits are added to slices
     *
     *  @param  pAlgorithm the address of the calling algorithm
     *  @param  caloHitListNames the hit type to calo hit list name map
     */
    virtual void InitializeHitLookup(const pandora::Algorithm *const pAlgorithm, const HitTypeToNameMap &caloHitListNames);

    /**
     *  @brief  Add the slice hits corresponding to a hit in a DL cluster
     *
     *  @param  pCaloHit the hit in the DL cluster
     *  @param  slice the slice to receive the hits
     */
    virtual void AddHitToSlice(const pandora::CaloHit *const pCaloHit, Slice &slice) const;

    std::string m_inputClusterListName; ///< The name of the input cluster list, with one cluster per slice
    std::string m_inputVertexListName;  ///< The name of the input vertex list
};

} // namespace lar_dl_content

#endif // #ifndef LAR_DL_EVENT_SLICING_TOOL_H
