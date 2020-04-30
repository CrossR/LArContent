/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.h
 *
 *  @brief  Implementation of the RANSAC related methods.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.h"

#include "larpandoracontent/LArUtility/RANSAC/PlaneModel.h"
#include "larpandoracontent/LArUtility/RANSAC/RANSAC.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

using namespace pandora;

namespace lar_content
{

void LArRANSACMethod::Run(ProtoHitVector &protoHitVector)
{
    ParameterVector candidatePoints;

    // TODO: Possibly want to sample this set for the initial RANSAC fit, just to speed it up.
    //       Having a cut off over X thousand hits perhaps?
    for (auto hit : m_consistentHits)
        candidatePoints.push_back(std::make_shared<Point3D>(hit));

    if (m_consistentHits.size() < 3)
        return; // TODO: Here we should default to the old behaviour.

    const float RANSAC_THRESHOLD = 2.5;
    const int RANSAC_ITERS = 100; // TODO: Should either be dynamic, or a config option.
    RANSAC<PlaneModel, 3> estimator(RANSAC_THRESHOLD, RANSAC_ITERS);
    estimator.Estimate(candidatePoints);

    m_parameterVectors.push_back(std::make_pair("bestInliers", estimator.GetBestInliers()));
    m_parameterVectors.push_back(std::make_pair("secondBestInliers", estimator.GetSecondBestInliers()));

    ProtoHitVector primaryResult;
    ProtoHitVector secondaryResult;
    int primaryModelCount = this->RunOverRANSACOutput(*estimator.GetBestModel(),
            estimator.GetBestInliers(), m_consistentHits, primaryResult,
            "best"
    );
    int secondModelCount = this->RunOverRANSACOutput(*estimator.GetSecondBestModel(),
            estimator.GetSecondBestInliers(), m_consistentHits, secondaryResult,
            "second"
    );

    int primaryTotal = estimator.GetBestInliers().size() + primaryModelCount;
    int secondaryTotal = estimator.GetSecondBestInliers().size() + secondModelCount;
    std::cout << primaryTotal << " vs " << secondaryTotal << std::endl;
    protoHitVector = primaryTotal > secondaryTotal ? primaryResult : secondaryResult;
}

int LArRANSACMethod::RunOverRANSACOutput(PlaneModel &currentModel, ParameterVector &currentInliers,
        ProtoHitVector &hitsToUse, ProtoHitVector &protoHitVector,
        std::string name
)
{
    std::map<const CaloHit*, std::pair<ProtoHit, float>> inlyingHitMap;
    const float RANSAC_THRESHOLD = 2.5; // TODO: Consolidate to config option.

    for (auto inlier : currentInliers)
    {
        auto hit = std::dynamic_pointer_cast<Point3D>(inlier);

        if (hit == nullptr)
            throw std::runtime_error("Inlying hit was not of type Point3D");

        LArRANSACMethod::AddToHitMap((*hit).m_ProtoHit, inlyingHitMap, 0.0);
    }

    if (currentInliers.size() < 1 || hitsToUse.size() < 1)
    {
        for (auto const& caloProtoPair : inlyingHitMap)
            protoHitVector.push_back(caloProtoPair.second.first);

        return 0;
    }

    ProtoHitVector nextHits;
    for (auto hit : hitsToUse)
        if ((inlyingHitMap.count(hit.GetParentCaloHit2D()) == 0))
            nextHits.push_back(hit);

    CartesianVector fitOrigin = currentModel.GetOrigin();
    CartesianVector fitDirection = currentModel.GetDirection();
    auto sortByModelDisplacement = [&fitOrigin, &fitDirection](ProtoHit a, ProtoHit b) {
        float displacementA = (a.GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        float displacementB = (b.GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        return displacementA < displacementB;
    };

    // Get the hits we will be using for the initial sliding fit.
    ProtoHitVector sortedHits;
    for (auto const& caloProtoPair : inlyingHitMap)
        sortedHits.push_back(caloProtoPair.second.first);

    if (sortedHits.size() < 3)
    {
        for (auto const& caloProtoPair : inlyingHitMap)
            protoHitVector.push_back(caloProtoPair.second.first);

        return 0;
    }

    std::sort(sortedHits.begin(), sortedHits.end(), sortByModelDisplacement);

    // TODO: If RANSAC was good enough (i.e. it got everything) skip this next bit.
    std::list<ProtoHit> currentPoints3D;

    for (auto hit : sortedHits)
        currentPoints3D.push_back(hit);

    std::cout << "Before iterations " << inlyingHitMap.size() << std::endl;

    const int FIT_ITERATIONS = 1000; // TODO: Config?

    ProtoHitVector hitsToUseForFit;
    std::vector<std::pair<ProtoHit, float>> hitsToAddToFit;

    LArRANSACMethod::GetHitsForFit(currentPoints3D, hitsToUseForFit, 0, 0);

    // Run fit from the start of the fit to the end an extend out.
    int smallIterCount = 0;
    ExtendDirection extendDirection = ExtendDirection::Forward;
    int coherentHitCount = 0;

    for (unsigned int iter = 0; iter < FIT_ITERATIONS; ++iter)
    {
        hitsToAddToFit.clear();

        for (float fits = 1; fits < 4.0; ++fits)
        {
            LArRANSACMethod::ExtendFit(
                nextHits, hitsToUseForFit, hitsToAddToFit,
                (RANSAC_THRESHOLD * fits), extendDirection,
                iter, name
            );

            if (hitsToAddToFit.size() > 0)
                break;
        }

        for (auto hitDispPair : hitsToAddToFit)
        {
            ++coherentHitCount;
            bool hitAdded = LArRANSACMethod::AddToHitMap(hitDispPair.first, inlyingHitMap, hitDispPair.second);
            if (hitAdded)
                hitsToUseForFit.push_back(hitDispPair.first);
        }

        bool continueFitting = LArRANSACMethod::GetHitsForFit(
                currentPoints3D, hitsToUseForFit, hitsToAddToFit.size(), smallIterCount
        );

        if (!continueFitting && extendDirection == ExtendDirection::Forward)
        {
            extendDirection = ExtendDirection::Backward;
            hitsToUseForFit.clear();
            currentPoints3D.clear();
            smallIterCount = 0;

            std::reverse(sortedHits.begin(), sortedHits.end());
            for (auto hit : sortedHits)
                currentPoints3D.push_back(hit);

            LArRANSACMethod::GetHitsForFit(currentPoints3D, hitsToUseForFit, 0, 0);
        }
        else if (!continueFitting && extendDirection == ExtendDirection::Backward)
            break;
    }

    for (auto const& caloProtoPair : inlyingHitMap)
        protoHitVector.push_back(caloProtoPair.second.first);

    m_allProtoHitsToPlot.push_back(std::make_pair("preIterativeHits_" + name, protoHitVector));
    m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_" + name, protoHitVector));

    const int totalCount = currentInliers.size() + coherentHitCount + inlyingHitMap.size();
    const int oldTotal = currentInliers.size() + coherentHitCount;
    std::cout << " RESULT: mapSize: " << inlyingHitMap.size() << ", HC: " << coherentHitCount << std::endl;
    std::cout << "         oldTotl: " << oldTotal << ", newTotal: " << totalCount << std::endl;

    return coherentHitCount;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool LArRANSACMethod::GetHitsForFit(
        std::list<ProtoHit> &currentPoints3D,
        ProtoHitVector &hitsToUseForFit,
        const int addedHitCount,
        int smallAdditionCount
)
{
    const int HITS_TO_KEEP = 80; // TODO: Config?
    const int FINISHED_THRESHOLD = 10; // TODO: Config

    // If we added no hits at the end, we should stop.
    if (addedHitCount == 0 && currentPoints3D.size() == 0)
        return false;

    // If we added a tiny number of hits for too many iterations, just
    // stop.
    if (addedHitCount < 2 && smallAdditionCount > FINISHED_THRESHOLD)
        return false;

    // If we added a tiny number of hits and are inside the fit, clear
    // the vector and move to the next N points.
    //
    // If we added a decent number of hits, just trim the points back
    // down to a reasonable size. This allows us to fill gaps in the
    // fit, and also extend out the end.
    if (addedHitCount <= 2 && currentPoints3D.size() != 0)
    {
        int i = 0;
        auto it = currentPoints3D.begin();
        while(i <= HITS_TO_KEEP && currentPoints3D.size() != 0)
        {
            hitsToUseForFit.push_back(*it);
            it = currentPoints3D.erase(it);

            if (hitsToUseForFit.size() >= HITS_TO_KEEP)
                hitsToUseForFit.erase(hitsToUseForFit.begin());

            ++i;
        }
    }
    else if (addedHitCount > 0)
    {
        auto it = hitsToUseForFit.begin();
        while (hitsToUseForFit.size() > HITS_TO_KEEP)
            it = hitsToUseForFit.erase(it);
    }

    // If we added a small number of hits at the end of the fit, start
    // counting to make sure we don't get stuck adding 1 hit repeatedly.
    //
    // On the other hand, if we recover a few iterations later and start
    // adding lots, reset the counter.
    if (addedHitCount < 5 && currentPoints3D.size() == 0)
        ++smallAdditionCount;
    else if (addedHitCount > 15)
        smallAdditionCount = 0;

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool LArRANSACMethod::AddToHitMap(
    ProtoHit hit,
    std::map<const CaloHit*, std::pair<ProtoHit, float>> &inlyingHitMap,
    float metricValue
)
{
    const CaloHit* twoDHit =  hit.GetParentCaloHit2D();

    if (inlyingHitMap.count(twoDHit) == 0)
    {
        inlyingHitMap[twoDHit] = std::make_pair(hit, metricValue);
        return true;
    }
    else
    {
        const float bestMetric = inlyingHitMap[twoDHit].second;
        if (metricValue < bestMetric)
        {
            inlyingHitMap[twoDHit] = std::make_pair(hit, metricValue);
            return true;
        }
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArRANSACMethod::ExtendFit(
    ProtoHitVector &hitsToTestAgainst, // TODO: List?
    ProtoHitVector &hitsToUseForFit,
    std::vector<std::pair<ProtoHit, float>> &hitsToAddToFit,
    const float distanceToFitThreshold,
    const ExtendDirection extendDirection,

    int iter, std::string name
)
{
    if (hitsToUseForFit.size() == 0)
        return;

    const float distanceToEndThreshold = 20; // TODO: Config.

    CartesianPointVector fitPoints;
    for (auto protoHit : hitsToUseForFit)
        fitPoints.push_back(protoHit.GetPosition3D());

    // TODO: Remove. Used for debugging.
    /*****************************************/
    ProtoHitVector hitsUsedInInitialFit;
    ProtoHitVector hitsAddedToFit;

    bool reverseFitDirection = extendDirection == ExtendDirection::Backward;

    for (auto protoHit : hitsToUseForFit)
    {
        ProtoHit newHit(protoHit.GetParentCaloHit2D());
        newHit.SetPosition3D(protoHit.GetPosition3D(), iter, reverseFitDirection);
        hitsUsedInInitialFit.push_back(newHit);
    }
    /*****************************************/

    // We've now got a sliding linear fit that should be based on the RANSAC fit.
    const unsigned int layerWindow(100); // TODO: May want this one to be different, since its for a different use.
    const ThreeDSlidingFitResult slidingFitResult(&fitPoints, layerWindow, m_pitch);
    std::cout << "     " << iter << "; Fit built using " << fitPoints.size() << " hits." << std::endl;

    CartesianVector fitDirection = slidingFitResult.GetGlobalMaxLayerDirection();
    CartesianVector fitEnd = slidingFitResult.GetGlobalMaxLayerPosition();

    if (extendDirection == ExtendDirection::Backward)
    {
        // TODO: This seems a bit iffy...does this work as I expect?
        fitDirection = slidingFitResult.GetGlobalMinLayerDirection();
        fitEnd = slidingFitResult.GetGlobalMinLayerPosition();
        fitDirection = fitDirection * -1.0;
    }

    int addedHits = 0;
    std::vector<int> hitsToPotentiallyCheck;

    // Use this sliding linear fit to test out the upcoming hits and pull some in
    int currentHitIndex = -1;
    auto it = hitsToTestAgainst.begin();
    while (it != hitsToTestAgainst.end())
    {
        // Get the position relative to the fit for the point.
        ++currentHitIndex;
        auto hit = *it;

        const CartesianVector pointPosition = hit.GetPosition3D();
        float dispFromFitEnd = (pointPosition - fitEnd).GetDotProduct(fitDirection);

        // If its not near the end of the fit, lets leave it to a subsequent iteration.
        if (dispFromFitEnd < 0.0 || std::abs(dispFromFitEnd) > distanceToEndThreshold)
        {
            ++it;
            continue;
        }

        CartesianVector projectedPosition(0.f, 0.f, 0.f);
        CartesianVector projectedDirection(0.f, 0.f, 0.f);
        const float rL(slidingFitResult.GetLongitudinalDisplacement(pointPosition));
        const StatusCode positionStatusCode(slidingFitResult.GetGlobalFitPosition(rL, projectedPosition));
        const StatusCode directionStatusCode(slidingFitResult.GetGlobalFitDirection(rL, projectedDirection));

        // Hits that failed to project but are still good are stored.
        // If we fail to project lots of hits, come back to these hits.
        if (positionStatusCode != STATUS_CODE_SUCCESS || directionStatusCode != STATUS_CODE_SUCCESS)
        {
            // We only use this when no hits are added, so iterator should be consistent.
            hitsToPotentiallyCheck.push_back(currentHitIndex);

            ++it;
            continue;
        }

        const float displacement = (pointPosition - projectedPosition).GetCrossProduct(projectedDirection).GetMagnitude();

        // If its good enough, lets store it and then we can pick the best one out later on.
        // Otherwise, just ignore it for now.
        if (displacement > distanceToFitThreshold)
        {
            ++it;
            continue;
        }

        // Since we are adding this hit, remove it from the hitsToTestAgainst vector.
        it = hitsToTestAgainst.erase(it);

        // TODO: Remove. Used for debugging.
        /*****************************************/
        ProtoHit newHit(hit.GetParentCaloHit2D());
        newHit.SetPosition3D(hit.GetPosition3D(), iter, reverseFitDirection);
        hitsAddedToFit.push_back(newHit);
        /*****************************************/

        ++addedHits;
        hitsToAddToFit.push_back(std::make_pair(hit, displacement));
    }

    // TODO: Remove. Used for debugging.
    /*****************************************/
    // std::cout << "############################################################################" << std::endl;
    // std::cout << "We used " << hitsToTestAgainst.size() << " hits in iteration..." << iter << std::endl;
    // std::cout << "We added " << addedHits << " hits in iteration..." << iter << std::endl;
    // std::cout << "There was " << hitsToPotentiallyCheck.size() << " hits to maybe use in iteration " << iter << "..." << std::endl;
    /*****************************************/

    if (addedHits != 0)
    {
        // std::cout << "############################################################################" << std::endl;
        m_allProtoHitsToPlot.push_back(std::make_pair("hitsUsedInFit_"    + name + "_" + std::to_string(iter), hitsUsedInInitialFit));
        m_allProtoHitsToPlot.push_back(std::make_pair("hitsAddedToFit_"   + name + "_" + std::to_string(iter), hitsAddedToFit));
        return;
    }

    // Now, lets instead just use the end of the fit to compare against.
    auto testIt = hitsToPotentiallyCheck.begin();
    while (testIt != hitsToPotentiallyCheck.end())
    {
        auto hit = hitsToTestAgainst[*testIt];
        CartesianVector pointPosition = hit.GetPosition3D();
        const float displacement = (pointPosition - fitEnd).GetCrossProduct(fitDirection).GetMagnitude();

        if (displacement > distanceToFitThreshold)
        {
            testIt = hitsToPotentiallyCheck.erase(testIt);
            continue;
        }

        ++testIt;
        ++addedHits;
        hitsToAddToFit.push_back(std::make_pair(hit, displacement));

        // TODO: Remove. Used for debugging.
        /*****************************************/
        ProtoHit newHit(hit.GetParentCaloHit2D());
        newHit.SetPosition3D(hit.GetPosition3D(), iter, reverseFitDirection);
        hitsAddedToFit.push_back(newHit);
        /*****************************************/
    }

    std::reverse(hitsToPotentiallyCheck.begin(), hitsToPotentiallyCheck.end());
    for (auto hitIndex : hitsToPotentiallyCheck)
    {
        auto hitIterator = std::next(hitsToTestAgainst.begin(), hitIndex);
        hitsToTestAgainst.erase(hitIterator);
    }

    /*****************************************/
    // std::cout << "We finally added " << addedHits << " hits in iteration " << iter << "..." << std::endl;
    // std::cout << "Avg displacement for fallback was " << sumOfDisplacements/float(addedHits) << " in iteration " << iter << "..." << std::endl;
    // std::cout << "############################################################################" << std::endl;
    m_allProtoHitsToPlot.push_back(std::make_pair("hitsUsedInFit_"    + name + "_" + std::to_string(iter), hitsUsedInInitialFit));
    m_allProtoHitsToPlot.push_back(std::make_pair("hitsAddedToFit_"   + name + "_" + std::to_string(iter), hitsAddedToFit));
    /*****************************************/
}
}
