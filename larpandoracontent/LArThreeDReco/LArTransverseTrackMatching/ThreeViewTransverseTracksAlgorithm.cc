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
    m_minSamplingPointsPerLayer(0.1f),
    m_cycleWindowSize(20),
    m_numZeroStates(10),
    m_minNumberOfCycles(5),
    m_minNumberOfCycleStates(10)
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

bool ThreeViewTransverseTracksAlgorithm::ShouldStopProcessing(std::vector<int> &changeHistory) const
{
    // Two ways we leave early here:
    // First, if there are was zero movement in the last m_numZeroStates iterations
    if (changeHistory.size() >= m_numZeroStates)
    {
        bool hasConverged{true};
        for (unsigned int i = 0; i < m_numZeroStates; ++i)
        {
            if (changeHistory.at(changeHistory.size() - 1 - i) != 0)
            {
                hasConverged = false;
                break;
            }
        }

        if (hasConverged)
        {
            std::cout << "Convergence detected: No changes in the last " << m_numZeroStates << " iterations." << std::endl;
            return true;
        }
    }
    else
        return false;

    // Secondly, are we stuck in some cyclic behaviour?
    //
    // That is, are we repeating some set of states over and over again?
    // Detect this by looking over a sliding window of the last m_cycleWindowSize states
    // looking for a repeating pattern of length 1 or more, with
    // thresholds on both the number of required cycles and the number of states in the cycle
    // sequence.
    unsigned int sequenceEnd(changeHistory.size() - 1);
    while (sequenceEnd > 0 && changeHistory[sequenceEnd] == 0)
        sequenceEnd--;

    // If the history is too short to have a meaningful cycle, return false
    if (sequenceEnd + 1 < m_minNumberOfCycleStates)
        return false;

    // Focus on the recent history (last X iterations or less if history is shorter)
    const unsigned int windowSize(std::min(m_cycleWindowSize, sequenceEnd + 1));

    for (unsigned int cycleLength = 1; cycleLength <= windowSize / 2; cycleLength++)
    {
        // Ensure we have enough history to reach the m_minNumberOfCycleStates with the current cycle length
        const unsigned int minCyclesForState = (m_minNumberOfCycleStates + cycleLength - 1) / cycleLength;
        const unsigned int requiredCycles = std::max(m_minNumberOfCycles, minCyclesForState);
        unsigned int confirmedCycles(0);

        // Check if we have enough history
        if (sequenceEnd + 1 < requiredCycles * cycleLength)
            continue;

        // Count how many consecutive, matching cycles exist at the end of the history
        for (unsigned int cycleIdx = 0; ; ++cycleIdx)
        {
            const unsigned int totalLength{ cycleLength * (cycleIdx + 1) };

            // Not enough history to check this cycle length
            if (sequenceEnd + 1 < totalLength)
                break;

            // Check if the current cycle matches the previous cycles
            bool isMatch{true};
            for (unsigned int i = 0; i < cycleLength; ++i)
            {
                if (changeHistory[sequenceEnd - i] != changeHistory[sequenceEnd - i - (cycleIdx * cycleLength)])
                {
                    isMatch = false;
                    break;
                }
            }

            // No match for this cycle length, stop checking further
            if (!isMatch)
                break;

            // Match found, increment confirmed cycles.
            confirmedCycles = cycleIdx + 1;
        }

        // Not enough confirmed cycles, skip to the next cycle length
        if (confirmedCycles < requiredCycles)
            continue;

        std::cout << "Cycle detected with period " << cycleLength << " (confirmed " << confirmedCycles << " cycles):" << std::endl;
        std::cout << "  Pattern: ";

        // Print the pattern
        for (unsigned int i = 0; i < cycleLength; i++)
        {
            std::cout << changeHistory[sequenceEnd - cycleLength + 1 + i];
            if (i < cycleLength - 1)
                std::cout << ", ";
        }
        std::cout << std::endl;

        // Found a valid cycle, return early
        return true;
    }

    // If we reach here, no cycles were detected
    std::cout << "No cycles detected in the last " << windowSize << " states." << std::endl;
    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ThreeViewTransverseTracksAlgorithm::ExamineOverlapContainer()
{
    unsigned int repeatCounter(0);

    std::vector<int> hitsMovedHistory;

    for (TensorToolVector::const_iterator iter = m_algorithmToolVector.begin(), iterEnd = m_algorithmToolVector.end(); iter != iterEnd;)
    {
        if ((*iter)->Run(this, this->GetMatchingControl().GetOverlapTensor()))
        {
            unsigned int nHitsMoved(this->GetModifiedHitCount());
            hitsMovedHistory.push_back(nHitsMoved);

            std::cout << "Iteration " << repeatCounter << " - moved hits: " << nHitsMoved << std::endl;

            if (++repeatCounter > m_nMaxTensorToolRepeats)
                break;

            if (ShouldStopProcessing(hitsMovedHistory))
                ++iter;
            else
                iter = m_algorithmToolVector.begin();

        }
        else
        {
            ++iter;
        }
    }

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

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "CycleWindowSize", m_cycleWindowSize));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "NumZeroStates", m_numZeroStates));

    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "MinNumberOfCycles", m_minNumberOfCycles));

    PANDORA_RETURN_RESULT_IF_AND_IF(
        STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle, "MinNumberOfCycleStates", m_minNumberOfCycleStates));

    return BaseAlgorithm::ReadSettings(xmlHandle);
}

} // namespace lar_content
