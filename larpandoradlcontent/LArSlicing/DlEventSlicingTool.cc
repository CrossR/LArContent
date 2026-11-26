/**
 *  @file   larpandoradlcontent/LArSlicing/DlEventSlicingTool.cc
 *
 *  @brief  Implementation of the DL event slicing tool class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"

#include "larpandoradlcontent/LArSlicing/DlEventSlicingTool.h"

using namespace pandora;
using namespace lar_content;

namespace lar_dl_content
{

DLEventSlicingTool::DLEventSlicingTool()
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DLEventSlicingTool::RunSlicing(const Algorithm *const pAlgorithm, const HitTypeToNameMap &caloHitListNames,
    const HitTypeToNameMap & /*clusterListNames*/, SliceList &sliceList)
{
    if (PandoraContentApi::GetSettings(*pAlgorithm)->ShouldDisplayAlgorithmInfo())
        std::cout << "----> Running Algorithm Tool: " << this->GetInstanceName() << ", " << this->GetType() << std::endl;

    // Get the clusters produced by the DL slicing, one per slice...
    const ClusterList *pClusterList(nullptr);
    PANDORA_THROW_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=,
        PandoraContentApi::GetList(*pAlgorithm, m_inputClusterListName, pClusterList));

    if (!pClusterList)
        throw StatusCodeException(STATUS_CODE_NOT_FOUND);

    // ...and the vertex seeds.
    const VertexList *pVertexList(nullptr);
    PANDORA_THROW_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_INITIALIZED, !=,
        PandoraContentApi::GetList(*pAlgorithm, m_inputVertexListName, pVertexList));

    if (!pVertexList)
        throw StatusCodeException(STATUS_CODE_NOT_FOUND);

    this->InitializeHitLookup(pAlgorithm, caloHitListNames);

    // Build up a map of slice index to the vertices that fall within that slice.
    std::map<unsigned int, VertexList> sliceIndexToVertexListMap;
    for (const Vertex *const pVertex : *pVertexList)
    {
        const CartesianVector &vertexPos(pVertex->GetPosition());

        // Find the first slice with a hit at the position of this vertex.
        bool vertexAssignedToSlice(false);
        unsigned int sliceIndex(0);
        for (const Cluster *const pCluster : *pClusterList)
        {
            CaloHitList caloHitList;
            LArClusterHelper::GetAllHits(pCluster, caloHitList);

            for (const CaloHit *const pCaloHit : caloHitList)
            {
                if (pCaloHit->GetPositionVector() == vertexPos)
                {
                    sliceIndexToVertexListMap[sliceIndex].push_back(pVertex);
                    vertexAssignedToSlice = true;
                    break;
                }
            }

            if (vertexAssignedToSlice)
                break;

            ++sliceIndex;
        }
    }

    // Create a slice for every cluster.
    for (const Cluster *const pCluster : *pClusterList)
    {
        Slice slice;

        CaloHitList caloHitList;
        LArClusterHelper::GetAllHits(pCluster, caloHitList);

        for (const CaloHit *const pCaloHit : caloHitList)
            this->AddHitToSlice(pCaloHit, slice);

        for (const Vertex *const pVertex : sliceIndexToVertexListMap[sliceList.size()])
            slice.m_vertexList.push_back(pVertex);

        sliceList.push_back(slice);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DLEventSlicingTool::InitializeHitLookup(const Algorithm *const /*pAlgorithm*/, const HitTypeToNameMap & /*caloHitListNames*/)
{
    // ATTN: This is empty for the 2D case, but is implemented in the native 3D slicing case.
}

//------------------------------------------------------------------------------------------------------------------------------------------

void DLEventSlicingTool::AddHitToSlice(const CaloHit *const pCaloHit, Slice &slice) const
{
    const HitType hitType(pCaloHit->GetHitType());

    if (TPC_VIEW_U == hitType)
        slice.m_caloHitListU.push_back(pCaloHit);
    else if (TPC_VIEW_V == hitType)
        slice.m_caloHitListV.push_back(pCaloHit);
    else if (TPC_VIEW_W == hitType)
        slice.m_caloHitListW.push_back(pCaloHit);
    else
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode DLEventSlicingTool::ReadSettings(const TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputClusterListName", m_inputClusterListName));
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "InputVertexListName", m_inputVertexListName));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_dl_content
