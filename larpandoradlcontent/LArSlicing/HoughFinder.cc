/**
 * @file   larpandoradlcontent/LArSlicing/HoughFinder.cc
 * @brief  Implementation of the Fast Hough Transform vertex finder
 */

#include <algorithm>
#include <cmath>
#include <numeric>

#include "larpandoradlcontent/LArSlicing/HoughFinder.h"
#include "larpandoradlcontent/LArSlicing/KnnKDTree.h"

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

    // Pre-compute bounds first.
    std::vector<float> voterLowerBoundSq(numVoters);
    std::vector<float> voterUpperBoundSq(numVoters);
    for (int v = 0; v < numVoters; ++v)
    {
        const float pd = predDists[voterIndices[v]];
        const float lb = std::max(0.0f, pd - m_tolerance);
        const float ub = pd + m_tolerance;
        voterLowerBoundSq[v] = lb * lb;
        voterUpperBoundSq[v] = ub * ub;
    }

    // Build a KD-Tree of all Candidates...
    // In testing, this was much faster.
    std::vector<lar_content::KnnKdTree::KnnNode> candNodes;
    candNodes.reserve(numCandidates);
    for (int c = 0; c < numCandidates; ++c)
    {
        const pandora::CartesianVector &pos = hitPositions[candidateIndices[c]];
        candNodes.push_back({{pos.GetX(), pos.GetY(), pos.GetZ()}, c});
    }

    lar_content::KnnKdTree candTree(candNodes);

    // Voters query the Candidate Tree using their specific upper bound.
    // I.e. using the distance class they have predicted.
    for (int v = 0; v < numVoters; ++v)
    {
        const int voterGlobalIdx = voterIndices[v];
        const pandora::CartesianVector &voterPos = hitPositions[voterGlobalIdx];

        lar_content::KnnKdTree::KnnNode queryNode = {{voterPos.GetX(), voterPos.GetY(), voterPos.GetZ()}, voterGlobalIdx};
        std::vector<int> closeCandidates = candTree.FindWithinRadiusSqd(queryNode, voterUpperBoundSq[v]);

        // For the candidates inside the upper bound, check the lower bound
        for (const int c : closeCandidates)
        {
            const float geomDistSq = (hitPositions[candidateIndices[c]] - voterPos).GetMagnitudeSquared();
            if (geomDistSq >= voterLowerBoundSq[v])
                voteCounts[c]++;
        }
    }

    // Sort the candidates based on a combined score of vote count and class (to
    // break ties in favor of more signal-like classes).
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
        const pandora::CartesianVector &candPos = hitPositions[candGlobalIdx];

        int currentSupport = 0;
        std::vector<int> claimedVotersLocal;

        for (int v = 0; v < numVoters; ++v)
        {
            if (!voterIsAvailable[v])
                continue;

            const int voterGlobalIdx = voterIndices[v];
            const float geomDistSq = (candPos - hitPositions[voterGlobalIdx]).GetMagnitudeSquared();

            if (geomDistSq >= voterLowerBoundSq[v] && geomDistSq <= voterUpperBoundSq[v])
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

//-----------------------------------------------------------------------------------------------------------------------------------------

} // namespace lar_content
