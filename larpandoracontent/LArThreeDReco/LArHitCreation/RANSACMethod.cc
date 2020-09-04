/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.h
 *
 *  @brief  Implementation of the RANSAC related methods.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"

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
        candidatePoints.push_back(std::make_shared<RANSACHit>(hit));

    if (m_consistentHits.size() < 3)
        return; // TODO: Here we should default to the old behaviour.

    const float RANSAC_THRESHOLD = 2.5 * 2.0;
    const int RANSAC_ITERS = 100; // TODO: Should either be dynamic, or a config option.
    RANSAC<PlaneModel, 3> estimator(RANSAC_THRESHOLD, RANSAC_ITERS);
    estimator.Estimate(candidatePoints);

    ProtoHitVector primaryResult;
    ProtoHitVector secondaryResult;
    m_name = "best"; // TODO: Remove;
    int primaryModelCount = this->RunOverRANSACOutput(estimator, RANSACResult::Best,
            m_consistentHits, primaryResult
    );
    m_name = "second"; // TODO: Remove;
    int secondModelCount = this->RunOverRANSACOutput(estimator, RANSACResult::Second,
            m_consistentHits, secondaryResult
    );

    const int primaryTotal = estimator.GetBestInliers().size() + primaryModelCount;
    const int secondaryTotal = estimator.GetSecondBestInliers().size() + secondModelCount;

    protoHitVector = primaryTotal > secondaryTotal ? primaryResult : secondaryResult;
}

int LArRANSACMethod::RunOverRANSACOutput(RANSAC<PlaneModel, 3> &ransac, RANSACResult run,
        RANSACHitVector &hitsToUse, ProtoHitVector &protoHitVector
)
{
    std::map<const CaloHit*, RANSACHit> inlyingHitMap;
    const float RANSAC_THRESHOLD = 1.0; // TODO: Consolidate to config option.

    const ParameterVector currentInliers = run == RANSACResult::Best ? ransac.GetBestInliers() : ransac.GetSecondBestInliers();

    if (currentInliers.size() == 0)
        return 0;

    /*****************************************/
    ProtoHitVector currentProtoHitInliers;
    for (auto inlier : currentInliers)
    {
        auto hit = *std::dynamic_pointer_cast<RANSACHit>(inlier);
        currentProtoHitInliers.push_back(hit.GetProtoHit());
    }
    m_allProtoHitsToPlot.push_back(std::make_pair(m_name + "Inliers", currentProtoHitInliers));
    /*****************************************/

    const auto bestModel = ransac.GetBestModel();
    auto secondModel = ransac.GetSecondBestModel();

    // ATTN: The second model can technically be NULL, so deal with that case.
    //       This can't happen for the main model, but we could stop before a
    //       second one. In this case, just set them to both be the same, since
    //       this doesn't affect anything.
    if (!secondModel)
        secondModel = bestModel;

    const PlaneModel currentModel = run == RANSACResult::Best ? *bestModel : *secondModel;
    const PlaneModel otherModel = run == RANSACResult::Best ? *secondModel : *bestModel;

    for (auto inlier : currentInliers)
    {
        auto hit = std::dynamic_pointer_cast<RANSACHit>(inlier);

        if (hit == nullptr)
            throw std::runtime_error("Inlying hit was not of type RANSACHit");

        LArRANSACMethod::AddToHitMap(*hit, inlyingHitMap);
    }

    if (currentInliers.size() < 1 || hitsToUse.size() < 1)
    {
        for (auto const& caloProtoPair : inlyingHitMap)
            protoHitVector.push_back(caloProtoPair.second.GetProtoHit());

        return 0;
    }

    std::list<RANSACHit> nextHits;
    for (auto hit : hitsToUse)
    {
        if ((inlyingHitMap.count(hit.GetProtoHit().GetParentCaloHit2D()) == 0))
        {
            // float modelDisp = otherModel.ComputeDistanceMeasure(std::make_shared<RANSACHit>(hit));

            // ATTN: A hit is unfavourable if its from a bad tool, or is in the
            //       other model. Unfavourable means it will attempt to not be
            //       used, but can be used if needed.
            // bool isNotInOtherModel = modelDisp >= RANSAC_THRESHOLD;
            bool isAlreadyFavourable = hit.IsFavourable();
            nextHits.push_back(RANSACHit(hit.GetProtoHit(), /*isNotInOtherModel && */ isAlreadyFavourable));
        }
    }

    const CartesianVector fitOrigin = currentModel.GetOrigin();
    const CartesianVector fitDirection = currentModel.GetDirection();
    auto sortByModelDisplacement = [&fitOrigin, &fitDirection] (RANSACHit a, RANSACHit b) {
        float displacementA = (a.GetProtoHit().GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        float displacementB = (b.GetProtoHit().GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        return displacementA < displacementB;
    };

    RANSACHitVector sortedHits;
    for (auto const& caloProtoPair : inlyingHitMap)
        sortedHits.push_back(caloProtoPair.second);

    if (sortedHits.size() < 3)
    {
        for (auto const& caloProtoPair : inlyingHitMap)
            protoHitVector.push_back(caloProtoPair.second.GetProtoHit());

        return 0;
    }

    std::sort(sortedHits.begin(), sortedHits.end(), sortByModelDisplacement);

    // TODO: If RANSAC was good enough (i.e. it got everything) skip this next bit.
    std::list<RANSACHit> currentPoints3D;

    for (auto hit : sortedHits)
        currentPoints3D.push_back(hit);

    const int FIT_ITERATIONS = 1000; // TODO: Config?

    RANSACHitVector hitsToUseForFit;
    RANSACHitVector hitsToAdd;

    LArRANSACMethod::GetHitsForFit(currentPoints3D, hitsToUseForFit, 0, 0);

    int smallIterCount = 0;
    ExtendDirection extendDirection = ExtendDirection::Forward;
    int coherentHitCount = 0;

    for (unsigned int iter = 0; iter < FIT_ITERATIONS; ++iter)
    {
        m_iter = iter;
        hitsToAdd.clear();

        for (float fits = 1; fits < 4.0; ++fits)
        {
            LArRANSACMethod::ExtendFit(
                nextHits, hitsToUseForFit, hitsToAdd,
                (RANSAC_THRESHOLD * fits), extendDirection
            );

            if (hitsToAdd.size() > 0)
                break;
        }

        for (auto hit : hitsToAdd)
        {
            bool hitAdded = LArRANSACMethod::AddToHitMap(hit, inlyingHitMap);

            if (hitAdded)
                hitsToUseForFit.push_back(hit);

            // TODO: Do we always want to add the hits? Or only at start/end, not the middle?
            //       Should this be apart of the other check above?
            if (currentPoints3D.size() == 0)
                currentPoints3D.push_back(hit);
        }

        const bool continueFitting =
            LArRANSACMethod::GetHitsForFit(currentPoints3D, hitsToUseForFit,
            hitsToAdd.size(), smallIterCount);
        coherentHitCount += hitsToAdd.size();

        // ATTN: If finished first pass, reset and reverse.
        //       If finished second (backwards) pass, stop.
        if (!continueFitting && extendDirection == ExtendDirection::Forward)
        {
            extendDirection = ExtendDirection::Backward;
            hitsToUseForFit.clear();
            currentPoints3D.clear();
            smallIterCount = 0;

            auto it = sortedHits.begin();
            // TODO: Randomly chosen "at least 5 fits worth", evaluate.
            while (currentPoints3D.size() < (5 * 80) && it != sortedHits.end())
            {
                currentPoints3D.push_back(*it);
                ++it;
            }

            std::reverse(currentPoints3D.begin(), currentPoints3D.end());
            LArRANSACMethod::GetHitsForFit(currentPoints3D, hitsToUseForFit, 0, 0);
        }
        else if (!continueFitting && extendDirection == ExtendDirection::Backward)
            break;
    }

    for (auto const& caloProtoPair : inlyingHitMap)
        protoHitVector.push_back(caloProtoPair.second.GetProtoHit());

    m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_" + m_name, protoHitVector));

    // ATTN: This count is the count of every considered hit, not the hits that were added.
    //       This allows distinguishing between models that fit to areas with more hits in.
    return coherentHitCount;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool LArRANSACMethod::GetHitsForFit(
        std::list<RANSACHit> &currentPoints3D,
        RANSACHitVector &hitsToUseForFit,
        const int addedHitCount,
        int smallAdditionCount
)
{
    const int HITS_TO_KEEP = 80; // TODO: Config and consolidate (reverse bit).
    const int FINISHED_THRESHOLD = 10; // TODO: Config

    // ATTN: Three options:
    //  Added no hits at the end: Stop.
    //  Added small number of hits at end repeatedly: Stop.
    //  Added small number of hits inside fit: Clear and add hits.
    //  Added lots of hits of hits: Don't add hits, just trim vector.
    if (addedHitCount == 0 && currentPoints3D.size() == 0)
        return false;

    if (addedHitCount < 2 && smallAdditionCount > FINISHED_THRESHOLD)
        return false;

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
        inlyingHitMap.insert({twoDHit, hit});
        return true;
    }
    else
    {
        const float bestMetric = inlyingHitMap.at(twoDHit).GetDisplacement();
        const float metricValue = hit.GetDisplacement();

        if (metricValue < bestMetric)
        {
            inlyingHitMap.insert({twoDHit, hit});
            return true;
        }
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool hitIsCloseToEnd(RANSACHit &hit, const CartesianVector &fitEnd,
        const CartesianVector &fitDirection, float threshold)
{
    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();
    const float dispFromFitEnd = (hitPosition - fitEnd).GetDotProduct(fitDirection);

    return (dispFromFitEnd > 0.0 && std::abs(dispFromFitEnd) < threshold);
}

void projectedHitDisplacement(RANSACHit &hit, const ThreeDSlidingFitResult slidingFit)
{
    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();

    CartesianVector projectedPosition(0.f, 0.f, 0.f);
    CartesianVector projectedDirection(0.f, 0.f, 0.f);
    const float rL(slidingFit.GetLongitudinalDisplacement(hitPosition));
    const StatusCode positionStatusCode(slidingFit.GetGlobalFitPosition(rL, projectedPosition));
    const StatusCode directionStatusCode(slidingFit.GetGlobalFitDirection(rL, projectedDirection));

    if (positionStatusCode != STATUS_CODE_SUCCESS || directionStatusCode != STATUS_CODE_SUCCESS)
        return;

    const float displacement = (hitPosition - projectedPosition).GetCrossProduct(projectedDirection).GetMagnitude();

    hit.SetDisplacement(displacement);
}

void hitDisplacement(RANSACHit &hit, const CartesianVector fitEnd, const CartesianVector fitDirection)
{
    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();
    const float displacement = (hitPosition - fitEnd).GetCrossProduct(fitDirection).GetMagnitude();

    hit.SetDisplacement(displacement);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArRANSACMethod::ExtendFit(
    std::list<RANSACHit> &hitsToTestAgainst,
    RANSACHitVector &hitsToUseForFit,
    std::vector<RANSACHit> &hitsToAdd,
    const float thresholdRatio,
    const ExtendDirection extendDirection
)
{
    if (hitsToUseForFit.size() == 0)
        return;

    float distanceToEndThreshold = 20;
    const float distanceToFitThreshold = 2.5 * thresholdRatio; // TODO: Config

    CartesianPointVector fitPoints;
    for (auto hit : hitsToUseForFit)
        fitPoints.push_back(hit.GetProtoHit().GetPosition3D());

    const unsigned int layerWindow(100);
    const ThreeDSlidingFitResult slidingFit(&fitPoints, layerWindow, m_pitch);

    CartesianVector fitDirection = slidingFit.GetGlobalMaxLayerDirection();
    CartesianVector fitEnd = slidingFit.GetGlobalMaxLayerPosition();

    if (extendDirection == ExtendDirection::Backward)
    {
        // TODO: This seems a bit iffy...does this work as I expect?
        fitDirection = slidingFit.GetGlobalMinLayerDirection();
        fitEnd = slidingFit.GetGlobalMinLayerPosition();
        fitDirection = fitDirection * -1.0;
    }

    // INFO: Check if there is a detector gap, and if there is, extend the fit over it.
    auto gapSize = LArGeometryHelper::ProjectAcrossGap3D(*m_pandora, fitEnd, fitDirection, 2.0, 10);
    if (gapSize > 0)
        distanceToEndThreshold += gapSize;

    // ATTN: This is done in 3 stages to split up the 3 different qualities of
    //       hits:
    //          - Hits based on the fit projections.
    //          - Hits based against the fit.
    //          - Unfavourable hits.
    auto projectedDisplacementTest = [&slidingFit] (RANSACHit &hit, float threshold) {
        projectedHitDisplacement(hit, slidingFit);
        if (hit.GetDisplacement() < threshold && hit.IsFavourable())
            return true;
        return false;
    };
    auto displacementTest = [&fitEnd, &fitDirection] (RANSACHit &hit, float threshold) {
        hitDisplacement(hit, fitEnd, fitDirection);
        if (hit.GetDisplacement() < threshold && hit.IsFavourable())
            return true;
        return false;
    };
    auto unfavourableTest = [] (RANSACHit &hit, float threshold) {
        if (hit.GetDisplacement() < threshold)
            return true;
        return false;
    };
    std::vector<std::function<bool(RANSACHit&, float)>> tests = {projectedDisplacementTest,
        displacementTest, unfavourableTest};

    /*****************************************/
    ProtoHitVector hitsComparedInFit;
    /*****************************************/

    std::vector<std::list<RANSACHit>::iterator> hitsToCheck;
    for (auto it = hitsToTestAgainst.begin(); it != hitsToTestAgainst.end(); ++it)
    {
        /*****************************************/
        const CartesianVector pointPosition = (*it).GetProtoHit().GetPosition3D();
        const float dispFromFitEnd = (pointPosition - fitEnd).GetDotProduct(fitDirection);

        ProtoHit newHit((*it).GetProtoHit().GetParentCaloHit2D());
        newHit.SetPosition3D((*it).GetProtoHit().GetPosition3D(), dispFromFitEnd, (*it).IsFavourable());
        hitsComparedInFit.push_back(newHit);
        /*****************************************/

        if (hitIsCloseToEnd((*it), fitEnd, fitDirection, distanceToEndThreshold))
        {
            hitsToCheck.push_back(it);
        }
    }

    unsigned int currentTest = 0;
    while (hitsToAdd.size() == 0 && currentTest < tests.size())
    {
        for (auto it : hitsToCheck)
        {
            if(tests[currentTest]((*it), distanceToFitThreshold))
            {
                hitsToAdd.push_back((*it));
                hitsToTestAgainst.erase(it);
            }
        }

        ++currentTest;
    }

    // // TODO: Remove. Used for debugging.
    // /*****************************************/
    // ProtoHitVector hitsUsedInInitialFit;
    // ProtoHitVector hitsAddedToFit;

    // bool reverseFitDirection = extendDirection == ExtendDirection::Backward;

    // for (auto hit : hitsToUseForFit) {
    //     ProtoHit newHit(hit.GetProtoHit().GetParentCaloHit2D());
    //     newHit.SetPosition3D(hit.GetProtoHit().GetPosition3D(), m_iter, reverseFitDirection);
    //     hitsUsedInInitialFit.push_back(newHit);
    // }
    // for (auto hit : hitsToAdd) {
    //     ProtoHit newHit(hit.GetProtoHit().GetParentCaloHit2D());
    //     newHit.SetPosition3D(hit.GetProtoHit().GetPosition3D(), m_iter, reverseFitDirection);
    //     hitsAddedToFit.push_back(newHit);
    // }
    // m_allProtoHitsToPlot.push_back(std::make_pair("hitsComparedInFit_"   + m_name + "_" + std::to_string(m_iter), hitsComparedInFit));
    // m_allProtoHitsToPlot.push_back(std::make_pair("hitsUsedInFit_"    + m_name + "_" + std::to_string(m_iter), hitsUsedInInitialFit));
    // m_allProtoHitsToPlot.push_back(std::make_pair("hitsAddedToFit_"   + m_name + "_" + std::to_string(m_iter), hitsAddedToFit));
    // /*****************************************/

    return;
}
}
