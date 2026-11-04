/**
 *  @file   larpandoracontent/LArTwoDReco/LArClusterCreation/SimpleClusterCreationAlgorithm.cc
 *
 *  @brief  Implementation of the cluster creation algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArHelpers/LArClusterHelper.h"

#include "larpandoracontent/LArTwoDReco/LArClusterCreation/SimpleClusterCreationAlgorithm.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

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
    const CaloHitList *pCaloHitList = NULL;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pCaloHitList));

    // Select available calo hits for clustering
    CaloHitList caloHitList;
    this->SelectCaloHits(pCaloHitList, caloHitList);

    if (caloHitList.empty())
        return STATUS_CODE_SUCCESS;

    // Build map of associations between selected calo hits
    HitAssociationMap hitAssociationMap;
    this->BuildAssociationMap(caloHitList, hitAssociationMap);

    // Create new clusters
    this->CreateClusters(caloHitList, hitAssociationMap);

    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SimpleClusterCreationAlgorithm::SelectCaloHits(const CaloHitList *const pInputList, CaloHitList &outputList) const
{
    for (const CaloHit *const pCaloHit : *pInputList)
    {
        if (PandoraContentApi::IsAvailable(*this, pCaloHit))
            outputList.push_back(pCaloHit);
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SimpleClusterCreationAlgorithm::BuildAssociationMap(const CaloHitList &caloHitList, HitAssociationMap &hitAssociationMap) const
{
    // Same result as the original all-pairs loop, but candidate neighbours come from a 2D (x,z) grid with cell size just above the
    // clustering window. The exact distance test and the resulting list order are unchanged: for hit I the list is
    // [first neighbour, I itself, remaining neighbours...], neighbours in input order (the original code looked up the list of I
    // when it meant J, which inserts I into its own list after its first neighbour; this is kept so the output is identical).
    const CaloHitVector hits(caloHitList.begin(), caloHitList.end());
    const std::size_t nHits(hits.size());

    const float cellSize(std::max(1.01f * std::sqrt(m_clusteringWindowSquared), std::numeric_limits<float>::min()));
    const auto cellIndex = [cellSize](const float value) { return static_cast<std::int64_t>(std::floor(value / cellSize)); };
    const auto cellKey = [](const std::int64_t ix, const std::int64_t iz) { return (ix * 2654435761LL) ^ (iz * 40503LL + (iz << 32)); };

    std::unordered_map<std::int64_t, std::vector<std::size_t>> grid;
    grid.reserve(nHits);
    std::vector<std::int64_t> ix(nHits), iz(nHits);

    for (std::size_t i = 0; i < nHits; ++i)
    {
        const CartesianVector &position(hits[i]->GetPositionVector());
        ix[i] = cellIndex(position.GetX());
        iz[i] = cellIndex(position.GetZ());
        grid[cellKey(ix[i], iz[i])].push_back(i); // indices ascending within a cell
    }

    std::vector<std::size_t> candidates;

    for (std::size_t i = 0; i < nHits; ++i)
    {
        candidates.clear();

        for (std::int64_t dx = -1; dx <= 1; ++dx)
        {
            for (std::int64_t dz = -1; dz <= 1; ++dz)
            {
                const auto iter(grid.find(cellKey(ix[i] + dx, iz[i] + dz)));

                if (iter != grid.end())
                    candidates.insert(candidates.end(), iter->second.begin(), iter->second.end());
            }
        }

        std::sort(candidates.begin(), candidates.end()); // restore input order
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end()); // cell-key collisions

        const CaloHit *const pCaloHitI(hits[i]);
        const CartesianVector &positionI(pCaloHitI->GetPositionVector());
        CaloHitList *pListI(nullptr);

        for (const std::size_t j : candidates)
        {
            const CaloHit *const pCaloHitJ(hits[j]);

            if (j == i || pCaloHitJ == pCaloHitI)
                continue;

            // Note: a hit pair appears at most once per I (distinct J), so the original duplicate-check never triggers for J
            if ((positionI - pCaloHitJ->GetPositionVector()).GetMagnitudeSquared() < m_clusteringWindowSquared)
            {
                if (!pListI)
                {
                    pListI = &hitAssociationMap[pCaloHitI];
                    pListI->push_back(pCaloHitJ);
                    pListI->push_back(pCaloHitI);
                }
                else
                {
                    pListI->push_back(pCaloHitJ);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SimpleClusterCreationAlgorithm::CreateClusters(const CaloHitList &caloHitList, const HitAssociationMap &hitAssociationMap) const
{
    CaloHitSet vetoList;
    CaloHitVector caloHitVector(caloHitList.begin(), caloHitList.end());
    std::sort(caloHitVector.begin(), caloHitVector.end(), LArClusterHelper::SortHitsByPosition);

    for (const CaloHit *const pSeedCaloHit : caloHitVector)
    {
        if (vetoList.count(pSeedCaloHit))
            continue;

        CaloHitList mergeList;
        this->CollectAssociatedHits(pSeedCaloHit, pSeedCaloHit, hitAssociationMap, vetoList, mergeList);

        const Cluster *pCluster = NULL;
        PandoraContentApi::Cluster::Parameters parameters;
        parameters.m_caloHitList.push_back(pSeedCaloHit);
        PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::Cluster::Create(*this, parameters, pCluster));
        vetoList.insert(pSeedCaloHit);

        for (const CaloHit *const pAssociatedCaloHit : mergeList)
        {
            PANDORA_THROW_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::AddToCluster(*this, pCluster, pAssociatedCaloHit));
            vetoList.insert(pAssociatedCaloHit);
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void SimpleClusterCreationAlgorithm::CollectAssociatedHits(const CaloHit *const pSeedCaloHit, const CaloHit *const pCurrentCaloHit,
    const HitAssociationMap &hitAssociationMap, const CaloHitSet &vetoList, CaloHitList &mergeList) const
{
    // Iterative depth-first traversal. Visits hits in exactly the same (pre-)order as the original recursive version
    // (neighbours sorted by position, appended to mergeList on first visit), but uses a hash set for the membership
    // test instead of a linear std::find over mergeList, and cannot overflow the stack on large clusters.
    struct Frame
    {
        CaloHitVector m_neighbours;
        std::size_t m_next;
    };

    CaloHitSet mergeSet(mergeList.begin(), mergeList.end());
    std::vector<Frame> stack;

    auto pushFrame = [&](const CaloHit *const pHit)
    {
        if (vetoList.count(pHit))
            return;

        HitAssociationMap::const_iterator iter = hitAssociationMap.find(pHit);
        if (iter == hitAssociationMap.end())
            return;

        stack.emplace_back();
        stack.back().m_neighbours.assign(iter->second.begin(), iter->second.end());
        stack.back().m_next = 0;
        std::sort(stack.back().m_neighbours.begin(), stack.back().m_neighbours.end(), LArClusterHelper::SortHitsByPosition);
    };

    pushFrame(pCurrentCaloHit);

    while (!stack.empty())
    {
        Frame &frame(stack.back());

        if (frame.m_next >= frame.m_neighbours.size())
        {
            stack.pop_back();
            continue;
        }

        const CaloHit *const pAssociatedCaloHit(frame.m_neighbours[frame.m_next++]);

        if (pAssociatedCaloHit == pSeedCaloHit)
            continue;

        if (!mergeSet.insert(pAssociatedCaloHit).second)
            continue;

        mergeList.push_back(pAssociatedCaloHit);
        pushFrame(pAssociatedCaloHit); // invalidates 'frame'
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode SimpleClusterCreationAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    float clusteringWindow = std::sqrt(m_clusteringWindowSquared);
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "ClusteringWindow", clusteringWindow));
    m_clusteringWindowSquared = clusteringWindow * clusteringWindow;

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
