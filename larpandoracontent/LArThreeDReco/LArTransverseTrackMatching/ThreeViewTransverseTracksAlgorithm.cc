/**
 *  @file   larpandoracontent/LArThreeDReco/LArTransverseTrackMatching/ThreeViewTransverseTracksAlgorithm.cc
 *
 *  @brief  Implementation of the three view transverse tracks algorithm class.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "Pandora/PandoraInternal.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArObjects/LArOverlapTensor.h"
#include "larpandoracontent/LArObjects/LArTrackOverlapResult.h"

#include "larpandoracontent/LArThreeDReco/LArTransverseTrackMatching/ThreeViewTransverseTracksAlgorithm.h"

using namespace pandora;

namespace lar_content
{

ThreeViewTransverseTracksAlgorithm::ThreeViewTransverseTracksAlgorithm() :
    m_nMaxTensorToolRepeats(1000),
    m_maxFitSegmentIndex(50),
    m_pseudoChi2Cut(3.f),
    m_minSegmentMatchedFraction(0.1f),
    m_minSegmentMatchedPoints(3),
    m_minOverallMatchedFraction(0.5f),
    m_minOverallMatchedPoints(10),
    m_minSamplingPointsPerLayer(0.1f)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeViewTransverseTracksAlgorithm::CalculateOverlapResult(const Cluster *const pClusterU, const Cluster *const pClusterV, const Cluster *const pClusterW)
{
    TransverseOverlapResult overlapResult;
    PANDORA_THROW_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, this->CalculateOverlapResult(pClusterU, pClusterV, pClusterW, overlapResult));

    if (overlapResult.IsInitialized())
        this->GetMatchingControl().GetOverlapTensor().SetOverlapResult(pClusterU, pClusterV, pClusterW, overlapResult);
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeViewTransverseTracksAlgorithm::CalculateOverlapResult(
    const Cluster *const pClusterU, const Cluster *const pClusterV, const Cluster *const pClusterW, TransverseOverlapResult &overlapResult)
{
    const TwoDSlidingFitResult &slidingFitResultU(this->GetCachedSlidingFitResult(pClusterU));
    const TwoDSlidingFitResult &slidingFitResultV(this->GetCachedSlidingFitResult(pClusterV));
    const TwoDSlidingFitResult &slidingFitResultW(this->GetCachedSlidingFitResult(pClusterW));

    FitSegmentTensor fitSegmentTensor;
    this->GetFitSegmentTensor(slidingFitResultU, slidingFitResultV, slidingFitResultW, fitSegmentTensor);

    TransverseOverlapResult transverseOverlapResult;
    this->GetBestOverlapResult(fitSegmentTensor, transverseOverlapResult);

    if (!transverseOverlapResult.IsInitialized())
        return STATUS_CODE_NOT_FOUND;

    if ((transverseOverlapResult.GetMatchedFraction() < m_minOverallMatchedFraction) ||
        (transverseOverlapResult.GetNMatchedSamplingPoints() < m_minOverallMatchedPoints))
        return STATUS_CODE_NOT_FOUND;

    const int nLayersSpannedU(slidingFitResultU.GetMaxLayer() - slidingFitResultU.GetMinLayer());
    const int nLayersSpannedV(slidingFitResultV.GetMaxLayer() - slidingFitResultV.GetMinLayer());
    const int nLayersSpannedW(slidingFitResultW.GetMaxLayer() - slidingFitResultW.GetMinLayer());
    const unsigned int meanLayersSpanned(
        static_cast<unsigned int>((1.f / 3.f) * static_cast<float>(nLayersSpannedU + nLayersSpannedV + nLayersSpannedW)));

    if (0 == meanLayersSpanned)
        return STATUS_CODE_FAILURE;

    const float nSamplingPointsPerLayer(static_cast<float>(transverseOverlapResult.GetNSamplingPoints()) / static_cast<float>(meanLayersSpanned));

    if (nSamplingPointsPerLayer < m_minSamplingPointsPerLayer)
        return STATUS_CODE_NOT_FOUND;

    overlapResult = transverseOverlapResult;
    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeViewTransverseTracksAlgorithm::GetFitSegmentTensor(const TwoDSlidingFitResult &slidingFitResultU,
    const TwoDSlidingFitResult &slidingFitResultV, const TwoDSlidingFitResult &slidingFitResultW, FitSegmentTensor &fitSegmentTensor) const
{
    FitSegmentList fitSegmentListU(slidingFitResultU.GetFitSegmentList());
    FitSegmentList fitSegmentListV(slidingFitResultV.GetFitSegmentList());
    FitSegmentList fitSegmentListW(slidingFitResultW.GetFitSegmentList());

    for (unsigned int indexU = 0, indexUEnd = fitSegmentListU.size(); indexU < indexUEnd; ++indexU)
    {
        const FitSegment &fitSegmentU(fitSegmentListU.at(indexU));

        for (unsigned int indexV = 0, indexVEnd = fitSegmentListV.size(); indexV < indexVEnd; ++indexV)
        {
            const FitSegment &fitSegmentV(fitSegmentListV.at(indexV));

            for (unsigned int indexW = 0, indexWEnd = fitSegmentListW.size(); indexW < indexWEnd; ++indexW)
            {
                const FitSegment &fitSegmentW(fitSegmentListW.at(indexW));

                TransverseOverlapResult segmentOverlap;
                if (STATUS_CODE_SUCCESS !=
                    this->GetSegmentOverlap(fitSegmentU, fitSegmentV, fitSegmentW, slidingFitResultU, slidingFitResultV, slidingFitResultW, segmentOverlap))
                    continue;

                if ((segmentOverlap.GetMatchedFraction() < m_minSegmentMatchedFraction) ||
                    (segmentOverlap.GetNMatchedSamplingPoints() < m_minSegmentMatchedPoints))
                    continue;

                fitSegmentTensor[indexU][indexV][indexW] = segmentOverlap;
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeViewTransverseTracksAlgorithm::GetSegmentOverlap(const FitSegment &fitSegmentU, const FitSegment &fitSegmentV,
    const FitSegment &fitSegmentW, const TwoDSlidingFitResult &slidingFitResultU, const TwoDSlidingFitResult &slidingFitResultV,
    const TwoDSlidingFitResult &slidingFitResultW, TransverseOverlapResult &transverseOverlapResult) const
{
    // Assess x-overlap
    const float xSpanU(fitSegmentU.GetMaxX() - fitSegmentU.GetMinX());
    const float xSpanV(fitSegmentV.GetMaxX() - fitSegmentV.GetMinX());
    const float xSpanW(fitSegmentW.GetMaxX() - fitSegmentW.GetMinX());

    const float minX(std::max(fitSegmentU.GetMinX(), std::max(fitSegmentV.GetMinX(), fitSegmentW.GetMinX())));
    const float maxX(std::min(fitSegmentU.GetMaxX(), std::min(fitSegmentV.GetMaxX(), fitSegmentW.GetMaxX())));
    const float xOverlap(maxX - minX);

    if (xOverlap < std::numeric_limits<float>::epsilon())
        return STATUS_CODE_NOT_FOUND;

    // Sampling in x
    const float nPointsU(std::fabs((xOverlap / xSpanU) * static_cast<float>(fitSegmentU.GetEndLayer() - fitSegmentU.GetStartLayer())));
    const float nPointsV(std::fabs((xOverlap / xSpanV) * static_cast<float>(fitSegmentV.GetEndLayer() - fitSegmentV.GetStartLayer())));
    const float nPointsW(std::fabs((xOverlap / xSpanW) * static_cast<float>(fitSegmentW.GetEndLayer() - fitSegmentW.GetStartLayer())));

    const unsigned int nPoints(1 + static_cast<unsigned int>((nPointsU + nPointsV + nPointsW) / 3.f));

    // Chi2 calculations
    float pseudoChi2Sum(0.f);
    unsigned int nSamplingPoints(0), nMatchedSamplingPoints(0);

    for (unsigned int n = 0; n <= nPoints; ++n)
    {
        const float x(minX + (maxX - minX) * static_cast<float>(n) / static_cast<float>(nPoints));

        CartesianVector fitUVector(0.f, 0.f, 0.f), fitVVector(0.f, 0.f, 0.f), fitWVector(0.f, 0.f, 0.f);
        CartesianVector fitUDirection(0.f, 0.f, 0.f), fitVDirection(0.f, 0.f, 0.f), fitWDirection(0.f, 0.f, 0.f);

        if ((STATUS_CODE_SUCCESS != slidingFitResultU.GetTransverseProjection(x, fitSegmentU, fitUVector, fitUDirection)) ||
            (STATUS_CODE_SUCCESS != slidingFitResultV.GetTransverseProjection(x, fitSegmentV, fitVVector, fitVDirection)) ||
            (STATUS_CODE_SUCCESS != slidingFitResultW.GetTransverseProjection(x, fitSegmentW, fitWVector, fitWDirection)))
        {
            continue;
        }

        const float u(fitUVector.GetZ()), v(fitVVector.GetZ()), w(fitWVector.GetZ());
        const float uv2w(LArGeometryHelper::MergeTwoPositions(this->GetPandora(), TPC_VIEW_U, TPC_VIEW_V, u, v));
        const float uw2v(LArGeometryHelper::MergeTwoPositions(this->GetPandora(), TPC_VIEW_U, TPC_VIEW_W, u, w));
        const float vw2u(LArGeometryHelper::MergeTwoPositions(this->GetPandora(), TPC_VIEW_V, TPC_VIEW_W, v, w));

        ++nSamplingPoints;
        const float deltaU((vw2u - u) * fitUDirection.GetX());
        const float deltaV((uw2v - v) * fitVDirection.GetX());
        const float deltaW((uv2w - w) * fitWDirection.GetX());

        const float pseudoChi2(deltaW * deltaW + deltaV * deltaV + deltaU * deltaU);
        pseudoChi2Sum += pseudoChi2;

        if (pseudoChi2 < m_pseudoChi2Cut)
            ++nMatchedSamplingPoints;
    }

    if (0 == nSamplingPoints)
        return STATUS_CODE_NOT_FOUND;

    const XOverlap xOverlapObject(fitSegmentU.GetMinX(), fitSegmentU.GetMaxX(), fitSegmentV.GetMinX(), fitSegmentV.GetMaxX(),
        fitSegmentW.GetMinX(), fitSegmentW.GetMaxX(), xOverlap);

    transverseOverlapResult = TransverseOverlapResult(nMatchedSamplingPoints, nSamplingPoints, pseudoChi2Sum, xOverlapObject);
    return STATUS_CODE_SUCCESS;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeViewTransverseTracksAlgorithm::GetBestOverlapResult(
    const FitSegmentTensor &fitSegmentTensor, TransverseOverlapResult &bestTransverseOverlapResult) const
{
    if (fitSegmentTensor.empty())
    {
        bestTransverseOverlapResult = TransverseOverlapResult();
        return;
    }

    unsigned int maxIndexU(0), maxIndexV(0), maxIndexW(0);
    for (FitSegmentTensor::const_iterator tIter = fitSegmentTensor.begin(), tIterEnd = fitSegmentTensor.end(); tIter != tIterEnd; ++tIter)
    {
        maxIndexU = std::max(tIter->first, maxIndexU);

        for (FitSegmentMatrix::const_iterator mIter = tIter->second.begin(), mIterEnd = tIter->second.end(); mIter != mIterEnd; ++mIter)
        {
            maxIndexV = std::max(mIter->first, maxIndexV);

            for (FitSegmentToOverlapResultMap::const_iterator rIter = mIter->second.begin(), rIterEnd = mIter->second.end(); rIter != rIterEnd; ++rIter)
                maxIndexW = std::max(rIter->first, maxIndexW);
        }
    }

    // ATTN Protect against longitudinal tracks winding back and forth; can cause large number of memory allocations
    maxIndexU = std::min(m_maxFitSegmentIndex, maxIndexU);
    maxIndexV = std::min(m_maxFitSegmentIndex, maxIndexV);
    maxIndexW = std::min(m_maxFitSegmentIndex, maxIndexW);

    FitSegmentTensor fitSegmentSumTensor;
    for (unsigned int indexU = 0; indexU <= maxIndexU; ++indexU)
    {
        for (unsigned int indexV = 0; indexV <= maxIndexV; ++indexV)
        {
            for (unsigned int indexW = 0; indexW <= maxIndexW; ++indexW)
            {
                TransverseOverlapResultVector transverseOverlapResultVector;
                this->GetPreviousOverlapResults(indexU, indexV, indexW, fitSegmentSumTensor, transverseOverlapResultVector);

                TransverseOverlapResultVector::const_iterator maxElement =
                    std::max_element(transverseOverlapResultVector.begin(), transverseOverlapResultVector.end());
                TransverseOverlapResult maxTransverseOverlapResult(
                    (transverseOverlapResultVector.end() != maxElement) ? *maxElement : TransverseOverlapResult());

                TransverseOverlapResult thisOverlapResult;
                if (fitSegmentTensor.count(indexU) && (fitSegmentTensor.find(indexU))->second.count(indexV) &&
                    ((fitSegmentTensor.find(indexU))->second.find(indexV))->second.count(indexW))
                    thisOverlapResult = (((fitSegmentTensor.find(indexU))->second.find(indexV))->second.find(indexW))->second;

                if (!thisOverlapResult.IsInitialized() && !maxTransverseOverlapResult.IsInitialized())
                    continue;

                fitSegmentSumTensor[indexU][indexV][indexW] = thisOverlapResult + maxTransverseOverlapResult;
            }
        }
    }

    for (unsigned int indexU = 0; indexU <= maxIndexU; ++indexU)
    {
        for (unsigned int indexV = 0; indexV <= maxIndexV; ++indexV)
        {
            for (unsigned int indexW = 0; indexW <= maxIndexW; ++indexW)
            {
                const TransverseOverlapResult &transverseOverlapResult(fitSegmentSumTensor[indexU][indexV][indexW]);

                if (!transverseOverlapResult.IsInitialized())
                    continue;

                if (transverseOverlapResult > bestTransverseOverlapResult)
                    bestTransverseOverlapResult = transverseOverlapResult;
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeViewTransverseTracksAlgorithm::GetPreviousOverlapResults(const unsigned int indexU, const unsigned int indexV,
    const unsigned int indexW, FitSegmentTensor &fitSegmentSumTensor, TransverseOverlapResultVector &transverseOverlapResultVector) const
{
    for (unsigned int iPermutation = 1; iPermutation < 8; ++iPermutation)
    {
        const bool decrementU((iPermutation >> 0) & 0x1);
        const bool decrementV((iPermutation >> 1) & 0x1);
        const bool decrementW((iPermutation >> 2) & 0x1);

        if ((decrementU && (0 == indexU)) || (decrementV && (0 == indexV)) || (decrementW && (0 == indexW)))
            continue;

        const unsigned int newIndexU(decrementU ? indexU - 1 : indexU);
        const unsigned int newIndexV(decrementV ? indexV - 1 : indexV);
        const unsigned int newIndexW(decrementW ? indexW - 1 : indexW);

        const TransverseOverlapResult &transverseOverlapResult(fitSegmentSumTensor[newIndexU][newIndexV][newIndexW]);

        if (transverseOverlapResult.IsInitialized())
            transverseOverlapResultVector.push_back(transverseOverlapResult);
    }

    if (transverseOverlapResultVector.empty())
        transverseOverlapResultVector.push_back(TransverseOverlapResult());
}

//------------------------------------------------------------------------------------------------------------------------------------------

int VisualiseTensorKeyClusters(const OverlapTensor<TransverseOverlapResult> &overlapTensor, const std::string &stateName,
    std::map<const Cluster *, std::map<std::pair<float, float>, const CaloHit *>> &previousStateMap)
{
    ClusterVector clusters;
    overlapTensor.GetSortedKeyClusters(clusters);
    std::cout << stateName << ") Number of clusters: " << clusters.size() << std::endl;

    std::map<const Cluster *, std::map<std::pair<float, float>, const CaloHit *>> stateMap;
    for (const auto &cluster : clusters)
    {
        CaloHitList caloHitList;
        cluster->GetOrderedCaloHitList().FillCaloHitList(caloHitList);
        for (const auto &caloHit : caloHitList)
        {
            const CartesianVector &position(caloHit->GetPositionVector());
            stateMap[cluster][std::make_pair(position.GetX(), position.GetZ())] = caloHit;
        }
    }

    if (previousStateMap.empty() && stateName != "InitialTensor")
    {
        std::cout << "No previous state map" << std::endl;
        previousStateMap = stateMap;
        return -1;
    }

    // Compare the two states, and print out a total of the hits that have moved
    unsigned int nMovedHits(0);

    for (const auto &entry : stateMap)
    {
        const Cluster *const pCluster(entry.first);
        const std::map<std::pair<float, float>, const CaloHit *> &caloHitMap(entry.second);

        if (!previousStateMap.count(pCluster))
        {
            nMovedHits += caloHitMap.size();
            continue;
        }

        const std::map<std::pair<float, float>, const CaloHit *> &previousCaloHitMap((previousStateMap)[pCluster]);

        for (const auto &caloHitEntry : caloHitMap)
        {
            const std::pair<float, float> &position(caloHitEntry.first);
            const CaloHit *const pCaloHit(caloHitEntry.second);

            if (!previousCaloHitMap.count(caloHitEntry.first))
            {
                ++nMovedHits;
                continue;
            }

            const CaloHit *const pPreviousCaloHit(previousCaloHitMap.at(position));

            if (pPreviousCaloHit != pCaloHit)
                ++nMovedHits;
        }
    }

    std::cout << "Number of moved hits: " << nMovedHits << std::endl;
    previousStateMap = stateMap;

    return nMovedHits;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeViewTransverseTracksAlgorithm::ExamineOverlapContainer()
{
    unsigned int repeatCounter(0);

    std::map<const Cluster *, std::map<std::pair<float, float>, const CaloHit *>> initialStateMap({});
    VisualiseTensorKeyClusters(this->GetMatchingControl().GetOverlapTensor(), "InitialTensor", initialStateMap);

    std::map<const Cluster *, std::map<std::pair<float, float>, const CaloHit *>> previousStateMap({});
    previousStateMap = initialStateMap;

    std::vector<int> hitsMovedHistory;

    for (TensorToolVector::const_iterator iter = m_algorithmToolVector.begin(), iterEnd = m_algorithmToolVector.end(); iter != iterEnd;)
    {
        if ((*iter)->Run(this, this->GetMatchingControl().GetOverlapTensor()))
        {
            iter = m_algorithmToolVector.begin();

            int nHitsMoved(VisualiseTensorKeyClusters(
                this->GetMatchingControl().GetOverlapTensor(), "Tensor_" + std::to_string(repeatCounter + 1), previousStateMap));
            hitsMovedHistory.push_back(nHitsMoved);

            if (++repeatCounter > m_nMaxTensorToolRepeats)
                break;

            // Two ways we leave early here:
            // 1) If the last 10 iterations have not moved any hits

            if (hitsMovedHistory.size() < 10)
                continue;

            int numZeroes(0);

            for (unsigned int i = 0; i < 10; ++i)
                if (hitsMovedHistory.at(hitsMovedHistory.size() - 1 - i) == 0)
                    ++numZeroes;

            if (numZeroes == 10)
                break;

            bool cyclic(false);

            // The second way is if the previous states are cyclic
            // Using a sliding window approach to detect recent cycles
            // Find the last non-zero value
            unsigned int sequenceEnd(hitsMovedHistory.size() - 1);
            while (sequenceEnd > 0 && hitsMovedHistory[sequenceEnd] == 0)
                sequenceEnd--;

            // Focus on the recent history (last 20 iterations or less if history is shorter)
            const unsigned int windowSize(std::min(20u, sequenceEnd + 1));

            for (unsigned int cycleLength = 1; cycleLength <= windowSize / 2; cycleLength++)
            {
                // Need at least enough elements to confirm pattern (minimum 10 total elements)
                unsigned int requiredCycles = std::max(5u, (10 / cycleLength + (5 % cycleLength > 0 ? 1 : 0)));

                // Check if we have enough history
                if (sequenceEnd + 1 < requiredCycles * cycleLength)
                    continue;

                // Check if the last 'cycleLength' elements match the previous 'cycleLength' elements
                bool potentialCycle(true);
                for (size_t i = 0; i < cycleLength; i++)
                {
                    // Compare current cycle with previous cycle
                    if (sequenceEnd - i < cycleLength || // Ensure we're not going out of bounds
                        hitsMovedHistory[sequenceEnd - i] != hitsMovedHistory[sequenceEnd - i - cycleLength])
                    {
                        potentialCycle = false;
                        break;
                    }
                }

                // Found a repeating pattern
                if (potentialCycle)
                {
                    // Try to extend the pattern validation backward to confirm it's really a cycle
                    size_t confirmedCycles = 2; // We already confirmed 2 cycles

                    // Check if earlier cycles also match
                    for (size_t extraCycle = 2; sequenceEnd + 1 >= (extraCycle + 1) * cycleLength; extraCycle++)
                    {
                        bool cycleMatches = true;

                        for (size_t i = 0; i < cycleLength; i++)
                        {
                            size_t currentPos = sequenceEnd - i;
                            size_t comparePos = sequenceEnd - i - (extraCycle * cycleLength);

                            if (comparePos >= hitsMovedHistory.size() || hitsMovedHistory[currentPos] != hitsMovedHistory[comparePos])
                            {
                                cycleMatches = false;
                                break;
                            }
                        }

                        if (cycleMatches)
                            confirmedCycles++;
                        else
                            break;
                    }

                    // Only consider it a real cycle if we find at least the required number of cycles
                    // and have at least 5 total elements in the pattern
                    if (confirmedCycles >= requiredCycles && (cycleLength * confirmedCycles >= 5))
                    {
                        std::cout << "Cycle detected with period " << cycleLength << " (confirmed " << confirmedCycles << " cycles):" << std::endl;
                        std::cout << "  Pattern: ";

                        // Print the pattern
                        for (size_t i = 0; i < cycleLength; i++)
                        {
                            std::cout << hitsMovedHistory[sequenceEnd - cycleLength + 1 + i];
                            if (i < cycleLength - 1)
                                std::cout << ", ";
                        }
                        std::cout << std::endl;

                        cyclic = true;
                        break;
                    }
                }
            }

            if (cyclic)
                break; // Exit the main loop if a cycle was detected
        }
        else
        {
            ++iter;
        }
    }

    std::cout << "Relative to initial state:" << std::endl;
    VisualiseTensorKeyClusters(this->GetMatchingControl().GetOverlapTensor(), "FinalTensor", initialStateMap);
    std::cout << "===> End of overlap tensor examination" << std::endl;
}

//------------------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode ThreeViewTransverseTracksAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
{
    AlgorithmToolVector algorithmToolVector;
    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ProcessAlgorithmToolList(*this, xmlHandle, "TrackTools", algorithmToolVector));

    for (AlgorithmToolVector::const_iterator iter = algorithmToolVector.begin(), iterEnd = algorithmToolVector.end(); iter != iterEnd; ++iter)
    {
        TransverseTensorTool *const pTransverseTensorTool(dynamic_cast<TransverseTensorTool *>(*iter));

        if (!pTransverseTensorTool)
            return STATUS_CODE_INVALID_PARAMETER;

        m_algorithmToolVector.push_back(pTransverseTensorTool);
    }

    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "NMaxTensorToolRepeats", m_nMaxTensorToolRepeats));

    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "MaxFitSegmentIndex", m_maxFitSegmentIndex));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "PseudoChi2Cut", m_pseudoChi2Cut));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadValue(xmlHandle, "MinSegmentMatchedFraction", m_minSegmentMatchedFraction));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadValue(xmlHandle, "MinSegmentMatchedPoints", m_minSegmentMatchedPoints));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadValue(xmlHandle, "MinOverallMatchedFraction", m_minOverallMatchedFraction));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadValue(xmlHandle, "MinOverallMatchedPoints", m_minOverallMatchedPoints));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=,
        XmlHelper::ReadValue(xmlHandle, "MinSamplingPointsPerLayer", m_minSamplingPointsPerLayer));

    return BaseAlgorithm::ReadSettings(xmlHandle);
}

} // namespace lar_content
