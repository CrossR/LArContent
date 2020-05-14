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

    protoHitVector = primaryTotal > secondaryTotal ? primaryResult : secondaryResult;
}

int LArRANSACMethod::RunOverRANSACOutput(PlaneModel &currentModel, ParameterVector &currentInliers,
        ProtoHitVector &hitsToUse, ProtoHitVector &protoHitVector,
        std::string name
)
{
    std::map<const CaloHit*, RANSACHit> inlyingHitMap;
    const float RANSAC_THRESHOLD = 2.5; // TODO: Consolidate to config option.

    for (auto inlier : currentInliers)
    {
        auto hit = std::dynamic_pointer_cast<Point3D>(inlier);

        if (hit == nullptr)
            throw std::runtime_error("Inlying hit was not of type Point3D");

        RANSACHit ransacHit((*hit).m_ProtoHit, 0.0);
        LArRANSACMethod::AddToHitMap(ransacHit, inlyingHitMap);
    }

    if (currentInliers.size() < 1 || hitsToUse.size() < 1)
    {
        for (auto const& caloProtoPair : inlyingHitMap)
            protoHitVector.push_back(caloProtoPair.second.GetProtoHit());

        return 0;
    }

    std::vector<RANSACHit> nextHits;
    for (auto hit : hitsToUse)
        if ((inlyingHitMap.count(hit.GetParentCaloHit2D()) == 0))
            nextHits.push_back(RANSACHit(hit, true)); // TODO: Set bool correctly.

    CartesianVector fitOrigin = currentModel.GetOrigin();
    CartesianVector fitDirection = currentModel.GetDirection();
    auto sortByModelDisplacement = [&fitOrigin, &fitDirection](ProtoHit a, ProtoHit b) {
        float displacementA = (a.GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        float displacementB = (b.GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        return displacementA < displacementB;
    };

    ProtoHitVector sortedHits;
    for (auto const& caloProtoPair : inlyingHitMap)
        sortedHits.push_back(caloProtoPair.second.GetProtoHit());

    if (sortedHits.size() < 3)
    {
        for (auto const& caloProtoPair : inlyingHitMap)
            protoHitVector.push_back(caloProtoPair.second.GetProtoHit());

        return 0;
    }

    std::sort(sortedHits.begin(), sortedHits.end(), sortByModelDisplacement);

    // TODO: If RANSAC was good enough (i.e. it got everything) skip this next bit.
    std::list<ProtoHit> currentPoints3D;

    for (auto hit : sortedHits)
        currentPoints3D.push_back(hit);

    const int FIT_ITERATIONS = 1000; // TODO: Config?

    ProtoHitVector hitsToUseForFit;
    std::vector<RANSACHit> hitsToAddToFit;

    LArRANSACMethod::GetHitsForFit(currentPoints3D, hitsToUseForFit, 0, 0);

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
                (RANSAC_THRESHOLD * fits), extendDirection
            );

            if (hitsToAddToFit.size() > 0)
                break;
        }

        for (auto hit : hitsToAddToFit)
        {
            bool hitAdded = LArRANSACMethod::AddToHitMap(hit, inlyingHitMap);

            if (hitAdded)
                hitsToUseForFit.push_back(hit.GetProtoHit());
        }

        const bool continueFitting =
            LArRANSACMethod::GetHitsForFit(currentPoints3D, hitsToUseForFit,
            hitsToAddToFit.size(), smallIterCount);
        coherentHitCount += hitsToAddToFit.size();

        // ATTN: If finished first pass, reset and reverse.
        //       If finished second (backwards) pass, stop.
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
        protoHitVector.push_back(caloProtoPair.second.GetProtoHit());

    m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_" + name, protoHitVector));

    // ATTN: This count is the count of every considered hit, not the hits that were added.
    //       This allows distinguishing between models that fit to areas with more hits in.
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

    // ATTN: This keeps track of the number of iterations that add only a small
    // number of hits. Resets if a larger iteration is reached.
    if (addedHitCount < 5 && currentPoints3D.size() == 0)
        ++smallAdditionCount;
    else if (addedHitCount > 15)
        smallAdditionCount = 0;

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool LArRANSACMethod::AddToHitMap(RANSACHit &hit, std::map<const CaloHit*, RANSACHit> &inlyingHitMap)
{
    const CaloHit* twoDHit =  hit.GetProtoHit().GetParentCaloHit2D();

    if (inlyingHitMap.count(twoDHit) == 0)
    {
        inlyingHitMap.insert(std::make_pair(twoDHit, hit));
        return true;
    }
    else
    {
        const float bestMetric = inlyingHitMap.at(twoDHit).GetDisplacement();
        const float metricValue = hit.GetDisplacement();

        if (metricValue < bestMetric)
        {
            inlyingHitMap.insert(std::make_pair(twoDHit, hit));
            return true;
        }
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool hitIsCloseToEnd(LArRANSACMethod::RANSACHit &hit, CartesianVector &fitEnd,
        CartesianVector &fitDirection, float threshold)
{
    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();
    const float dispFromFitEnd = (hitPosition - fitEnd).GetDotProduct(fitDirection);

    return (dispFromFitEnd > 0.0 && dispFromFitEnd < threshold);
}

void projectedHitDisplacement(LArRANSACMethod::RANSACHit hit, ThreeDSlidingFitResult slidingFit)
{

    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();

    CartesianVector projectedPosition(0.f, 0.f, 0.f);
    CartesianVector projectedDirection(0.f, 0.f, 0.f);
    const float rL(slidingFit.GetLongitudinalDisplacement(hitPosition));
    const StatusCode positionStatusCode(slidingFit.GetGlobalFitPosition(rL, projectedPosition));
    const StatusCode directionStatusCode(slidingFit.GetGlobalFitDirection(rL, projectedDirection));

    // ATTN: If the hit failed to project, store it in case we have to fallback to the simpler method.
    if (positionStatusCode != STATUS_CODE_SUCCESS || directionStatusCode != STATUS_CODE_SUCCESS)
        return;

    const float displacement = (hitPosition - projectedPosition).GetCrossProduct(projectedDirection).GetMagnitude();

    hit.SetDisplacement(displacement);
}

void hitDisplacement(LArRANSACMethod::RANSACHit hit, CartesianVector fitEnd, CartesianVector fitDirection)
{
    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();
    const float displacement = (hitPosition - fitEnd).GetCrossProduct(fitDirection).GetMagnitude();

    hit.SetDisplacement(displacement);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArRANSACMethod::ExtendFit(
    std::vector<RANSACHit> &hitsToTestAgainst, // TODO: List?
    ProtoHitVector &hitsToUseForFit,
    std::vector<RANSACHit> &hitsToAddToFit,
    const float distanceToFitThreshold,
    const ExtendDirection extendDirection
)
{
    if (hitsToUseForFit.size() == 0)
        return;

    const float distanceToEndThreshold = 20; // TODO: Config.

    CartesianPointVector fitPoints;
    for (auto protoHit : hitsToUseForFit)
        fitPoints.push_back(protoHit.GetPosition3D());

    const unsigned int layerWindow(100);
    const ThreeDSlidingFitResult slidingFitResult(&fitPoints, layerWindow, m_pitch);

    CartesianVector fitDirection = slidingFitResult.GetGlobalMaxLayerDirection();
    CartesianVector fitEnd = slidingFitResult.GetGlobalMaxLayerPosition();

    if (extendDirection == ExtendDirection::Backward)
    {
        // TODO: This seems a bit iffy...does this work as I expect?
        fitDirection = slidingFitResult.GetGlobalMinLayerDirection();
        fitEnd = slidingFitResult.GetGlobalMinLayerPosition();
        fitDirection = fitDirection * -1.0;
    }

    std::vector<RANSACHit> hitsToCheck;

    // ATTN: This is done in 3 stages to split up the 3 different qualities of
    //       hits:
    //          - Hits based on the fit projections.
    //          - Hits based against the fit.
    //          - Unfavourable hits.
    for (auto hit : hitsToTestAgainst)
    {
        if (!hitIsCloseToEnd(hit, fitEnd, fitDirection, distanceToEndThreshold))
            continue;

        projectedHitDisplacement(hit, slidingFitResult);

        if (hit.GetDisplacement() < distanceToFitThreshold && hit.IsFavourable())
            hitsToAddToFit.push_back(hit);
        else
            hitsToCheck.push_back(hit);
    }

    if (hitsToAddToFit.size() > 0)
        return;

    for (auto hit : hitsToCheck)
    {
        hitDisplacement(hit, fitEnd, fitDirection);

        if (hit.GetDisplacement() < distanceToFitThreshold && hit.IsFavourable())
            hitsToAddToFit.push_back(hit);
    }

    if (hitsToAddToFit.size() > 0)
        return;

    for (auto hit : hitsToCheck)
    {
        if (hit.GetDisplacement() < distanceToFitThreshold)
            hitsToAddToFit.push_back(hit);
    }

    // TODO: Remove hits

    return;
}
}
