/**
 * @file   larpandoradlcontent/LArSlicing/HoughFinder.cc
 * @brief  Implementation of the Fast Hough Transform vertex finder
 */

#include <algorithm>
#include <cmath>
#include <numeric>

#include "larpandoradlcontent/LArSlicing/HoughFinder.h"

namespace lar_content
{

FastHoughFinder::FastHoughFinder(
    const std::vector<float> &thresholds, const float scalingFactor, const float tolerance, const int minVotes, const float nmsRadius) :
    m_thresholds(thresholds),
    m_scalingFactor(scalingFactor),
    m_tolerance(tolerance),
    m_minVotes(minVotes),
    m_nmsRadiusSq(nmsRadius * nmsRadius)
{
    float prev_t = 0.0f;
    for (const float t : m_thresholds)
    {
        m_binCenters.push_back((prev_t + t) / 2.0f);
        prev_t = t;
    }
    m_binCenters.push_back(0.0f);
}

std::vector<pandora::CartesianVector> FastHoughFinder::Fit(const std::vector<pandora::CartesianVector> &hitPositions, const std::vector<float> &logits) const
{
    const int numHits = hitPositions.size();
    const int numClasses = m_thresholds.size() + 1;

    if (numHits == 0)
        return {};

    std::vector<float> predDists(numHits);
    std::vector<int> candidateIndices;
    std::vector<int> voterIndices;

    for (int i = 0; i < numHits; ++i)
    {
        int bestClass = 0;
        float maxLogit = logits[i * numClasses];
        for (int c = 1; c < numClasses; ++c)
        {
            const float val = logits[i * numClasses + c];
            if (val > maxLogit)
            {
                maxLogit = val;
                bestClass = c;
            }
        }
        predDists[i] = m_binCenters[bestClass] * m_scalingFactor;

        if (bestClass <= 2)
        {
            candidateIndices.push_back(i);
        }

        if (bestClass != numClasses - 1)
        {
            voterIndices.push_back(i);
        }
    }

    if (candidateIndices.empty())
        return {};

    const int numCandidates = candidateIndices.size();
    const int numVoters = voterIndices.size();
    std::vector<int> voteCounts(numCandidates, 0);

    for (int c = 0; c < numCandidates; ++c)
    {
        const pandora::CartesianVector &candPos = hitPositions[candidateIndices[c]];
        int votes = 0;

        for (int v = 0; v < numVoters; ++v)
        {
            const int voterGlobalIdx = voterIndices[v];
            const float geomDist = (candPos - hitPositions[voterGlobalIdx]).GetMagnitude();

            if (std::abs(geomDist - predDists[voterGlobalIdx]) < m_tolerance)
            {
                votes++;
            }
        }
        voteCounts[c] = votes;
    }

    std::vector<int> sortedCandIndices(numCandidates);
    std::iota(sortedCandIndices.begin(), sortedCandIndices.end(), 0);
    std::sort(sortedCandIndices.begin(), sortedCandIndices.end(), [&voteCounts](int i1, int i2) { return voteCounts[i1] > voteCounts[i2]; });

    std::vector<pandora::CartesianVector> foundVertices;
    std::vector<bool> candidateIsAvailable(numCandidates, true);
    std::vector<bool> voterIsAvailable(numVoters, true);

    for (const int candListIdx : sortedCandIndices)
    {
        if (!candidateIsAvailable[candListIdx])
            continue;
        if (voteCounts[candListIdx] < m_minVotes)
            break;

        const int candGlobalIdx = candidateIndices[candListIdx];
        const pandora::CartesianVector &candPos = hitPositions[candGlobalIdx];

        int currentSupport = 0;
        std::vector<int> claimedVotersLocal;

        for (int v = 0; v < numVoters; ++v)
        {
            if (!voterIsAvailable[v])
                continue;

            const int voterGlobalIdx = voterIndices[v];
            const float geomDist = (candPos - hitPositions[voterGlobalIdx]).GetMagnitude();

            if (std::abs(geomDist - predDists[voterGlobalIdx]) < m_tolerance)
            {
                currentSupport++;
                claimedVotersLocal.push_back(v);
            }
        }

        if (currentSupport < m_minVotes)
            continue;

        foundVertices.push_back(candPos);

        // Consume the voters so they can't vote for subsequent nearby candidates
        for (const int localVoterIdx : claimedVotersLocal)
        {
            voterIsAvailable[localVoterIdx] = false;
        }

        for (int c = 0; c < numCandidates; ++c)
        {
            if (!candidateIsAvailable[c])
                continue;

            const pandora::CartesianVector &otherCandPos = hitPositions[candidateIndices[c]];
            if ((candPos - otherCandPos).GetMagnitudeSquared() < m_nmsRadiusSq)
            {
                candidateIsAvailable[c] = false;
            }
        }
    }

    return foundVertices;
}

} // namespace lar_content