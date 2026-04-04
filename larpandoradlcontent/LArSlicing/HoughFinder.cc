/**
 * @file   larpandoradlcontent/LArSlicing/HoughFinder.cc
 * @brief  Implementation of the Fast Hough Transform vertex finder
 */

#include <algorithm>
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

//-----------------------------------------------------------------------------------------------------------------------------------------

std::vector<pandora::CartesianVector> FastHoughFinder::Fit(const std::vector<pandora::CartesianVector> &hitPositions, const std::vector<float> &logits) const
{
    const int numHits = hitPositions.size();
    const int numClasses = m_thresholds.size() + 1;

    if (numHits == 0)
        return {};

    std::vector<float> predDists(numHits);
    std::vector<int> predClasses(numHits);
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
        predClasses[i] = bestClass;

        // TODO: Param for classes as candidates vs voters.
        if (bestClass <= 2)
            candidateIndices.push_back(i);

        // INFO: The last class is noise, so we don't want those voting.
        if (bestClass != numClasses - 1)
            voterIndices.push_back(i);
    }

    if (candidateIndices.empty())
        return {};

    const int numCandidates = candidateIndices.size();
    const int numVoters = voterIndices.size();
    std::vector<int> voteCounts(numCandidates, 0);
    std::vector<int> sortScores(numCandidates, 0);

    // Simple vectors for positions, to help with cache locality.
    std::vector<float> vX(numVoters), vY(numVoters), vZ(numVoters);
    std::vector<float> voterLowerBoundSq(numVoters), voterUpperBoundSq(numVoters);

    // Precompute the voter positions and their distance bounds for voting.
    for (int v = 0; v < numVoters; ++v)
    {
        const int idx = voterIndices[v];
        const pandora::CartesianVector &pos = hitPositions[idx];

        vX[v] = pos.GetX();
        vY[v] = pos.GetY();
        vZ[v] = pos.GetZ();

        const float pd = predDists[idx];
        const float lb = std::max(0.0f, pd - m_tolerance);
        const float ub = pd + m_tolerance;
        voterLowerBoundSq[v] = lb * lb;
        voterUpperBoundSq[v] = ub * ub;
    }

    // Voting loop - for each candidate, count how many voters are within the
    // predicted distance (with tolerance).
    for (int c = 0; c < numCandidates; ++c)
    {
        const int candIdx = candidateIndices[c];
        const float cX = hitPositions[candIdx].GetX();
        const float cY = hitPositions[candIdx].GetY();
        const float cZ = hitPositions[candIdx].GetZ();

        int votes = 0;

        for (int v = 0; v < numVoters; ++v)
        {
            const float dx = cX - vX[v];
            const float dy = cY - vY[v];
            const float dz = cZ - vZ[v];
            const float distSq = (dx * dx) + (dy * dy) + (dz * dz);

            votes += (distSq >= voterLowerBoundSq[v]) & (distSq <= voterUpperBoundSq[v]);
        }

        voteCounts[c] = votes;
    }

    // Calculate the sort scores
    for (int c = 0; c < numCandidates; ++c)
    {
        const int candClass = predClasses[candidateIndices[c]];
        sortScores[c] = voteCounts[c] + ((3 - candClass) * 100);
    }

    std::vector<int> sortedCandIndices(numCandidates);
    std::iota(sortedCandIndices.begin(), sortedCandIndices.end(), 0);
    std::sort(sortedCandIndices.begin(), sortedCandIndices.end(), [&sortScores](int a, int b) { return sortScores[a] > sortScores[b]; });

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
        const float cX = hitPositions[candGlobalIdx].GetX();
        const float cY = hitPositions[candGlobalIdx].GetY();
        const float cZ = hitPositions[candGlobalIdx].GetZ();

        int currentSupport = 0;
        std::vector<int> claimedVotersLocal;

        for (int v = 0; v < numVoters; ++v)
        {
            if (!voterIsAvailable[v])
                continue;

            const float dx = cX - vX[v];
            const float dy = cY - vY[v];
            const float dz = cZ - vZ[v];
            const float geomDistSq = (dx * dx) + (dy * dy) + (dz * dz);

            if (geomDistSq >= voterLowerBoundSq[v] && geomDistSq <= voterUpperBoundSq[v])
            {
                currentSupport++;
                claimedVotersLocal.push_back(v);
            }
        }

        if (currentSupport < m_minVotes)
            continue;

        const auto candPos = pandora::CartesianVector(cX, cY, cZ);
        foundVertices.push_back(candPos);

        // Consume the voters so they can't vote for subsequent nearby candidates
        for (const int localVoterIdx : claimedVotersLocal)
            voterIsAvailable[localVoterIdx] = false;

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

//-----------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_content
