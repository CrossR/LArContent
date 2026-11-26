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

private:
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    std::string m_inputVertexListName; ///< The name of the input vertex list
};

} // namespace lar_dl_content

#endif // #ifndef LAR_DL_EVENT_SLICING_TOOL_H
