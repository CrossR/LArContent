/**
 *  @file   larpandoracontent/LArTwoDReco/LArClusterCreation/SimpleClusterCreationAlgorithm.cc
 *
 *  @brief  Implementation of the cluster creation algorithm classes.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArUtility/KDTreeLinkerToolsT.h"

#include "larpandoracontent/LArTwoDReco/LArClusterCreation/SimpleClusterCreationAlgorithm.h"

using namespace pandora;

namespace lar_content
{

SimpleClusterCreationAlgorithm::SimpleClusterCreationAlgorithm() :
    m_clusteringWindowSquared(1.f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SimpleClusterCreationAlgorithm::Run()
{
    const CaloHitList *pCaloHitList(nullptr);

    if (m_inputCaloHitListName.empty())
    {
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pCaloHitList));
    }
    else
    {
        PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_inputCaloHitListName, pCaloHitList));
    }

    CaloHitVector caloHitVector;

    for (const CaloHit *const pCaloHit : *pCaloHitList)
    {
        if (PandoraContentApi::IsAvailable(*this, pCaloHit))
            caloHitVector.push_back(pCaloHit);
    }

    if (caloHitVector.empty())
        return STATUS_CODE_SUCCESS;

    std::sort(caloHitVector.begin(), caloHitVector.end(), LArClusterHelper::SortHitsByPosition);

    HitAssociationMap hitAssociationMap;
    this->BuildAssociationMap(caloHitVector, hitAssociationMap);
    this->CreateClusters(caloHitVector, hitAssociationMap);

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SimpleClusterCreationAlgorithm::BuildAssociationMap(const CaloHitVector &caloHitVector, HitAssociationMap &hitAssociationMap) const
{
    typedef KDTreeLinkerAlgo<const CaloHit *, 2> HitKDTree;
    typedef KDTreeNodeInfoT<const CaloHit *, 2> HitKDNode;

    const CaloHitList caloHitList(caloHitVector.begin(), caloHitVector.end());
    std::vector<HitKDNode> hitKDNodes;
    HitKDTree kdTree;
    kdTree.build(hitKDNodes, fill_and_bound_2d_kd_tree(caloHitList, hitKDNodes));

    const float window(std::sqrt(m_clusteringWindowSquared));

    for (const CaloHit *const pCaloHitI : caloHitVector)
    {
        std::vector<HitKDNode> found;
        kdTree.search(build_2d_kd_search_region(pCaloHitI, window, window), found);

        CaloHitVector nearbyHits;

        for (const HitKDNode &node : found)
        {
            const CaloHit *const pCaloHitJ(node.data);

            if (pCaloHitJ != pCaloHitI && (pCaloHitI->GetPositionVector() - pCaloHitJ->GetPositionVector()).GetMagnitudeSquared() < m_clusteringWindowSquared)
                nearbyHits.push_back(pCaloHitJ);
        }

        if (!nearbyHits.empty())
        {
            std::sort(nearbyHits.begin(), nearbyHits.end(), LArClusterHelper::SortHitsByPosition);
            hitAssociationMap.emplace(pCaloHitI, std::move(nearbyHits));
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SimpleClusterCreationAlgorithm::CreateClusters(const CaloHitVector &caloHitVector, const HitAssociationMap &hitAssociationMap) const
{
    if (!m_outputClusterListName.empty())
    {
        const ClusterList *pClusterList(nullptr);
        std::string temporaryListName;
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::CreateTemporaryListAndSetCurrent(*this, pClusterList, temporaryListName));
    }

    CaloHitSet usedHits;

    for (const CaloHit *const pSeedCaloHit : caloHitVector)
    {
        if (!usedHits.insert(pSeedCaloHit).second)
            continue;

        PandoraContentApi::Cluster::Parameters parameters;
        parameters.m_caloHitList.push_back(pSeedCaloHit);

        std::vector<std::pair<const CaloHitVector *, std::size_t>> stack;

        // INFO: Lambda function to push nearby hits onto the stack, if they
        // exist in the association map
        //
        // This is needed because the references to the nearby hit vectors are
        // invalidated when the vector is moved into the map, so we can't just
        // iterate over the map directly
        const auto push = [&](const CaloHit *const pCaloHit)
        {
            const auto iter(hitAssociationMap.find(pCaloHit));

            if (iter != hitAssociationMap.end())
                stack.emplace_back(&iter->second, 0);
        };

        push(pSeedCaloHit);

        while (!stack.empty())
        {
            auto &[pNearbyHits, nextIndex] = stack.back();

            if (nextIndex == pNearbyHits->size())
            {
                stack.pop_back();
                continue;
            }

            const CaloHit *const pNearbyHit((*pNearbyHits)[nextIndex++]);

            if (usedHits.insert(pNearbyHit).second)
            {
                parameters.m_caloHitList.push_back(pNearbyHit);
                push(pNearbyHit);
            }
        }

        const Cluster *pCluster(nullptr);
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, parameters, pCluster));
    }

    if (!m_outputClusterListName.empty())
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::SaveList<Cluster>(*this, m_outputClusterListName));
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SimpleClusterCreationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    float clusteringWindow = std::sqrt(m_clusteringWindowSquared);
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "ClusteringWindow", clusteringWindow));
    m_clusteringWindowSquared = clusteringWindow * clusteringWindow;

    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "InputCaloHitListName", m_inputCaloHitListName));
    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "OutputClusterListName", m_outputClusterListName));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
