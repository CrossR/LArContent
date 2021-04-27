/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.cc
 *
 *  @brief  Implementation of the RANSAC related methods.
 *
 *  $Log: $
 */

#include "Pandora/AlgorithmHeaders.h"

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "larpandoracontent/LArHelpers/LArRandomHelper.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.h"

#include "larpandoracontent/LArUtility/RANSAC/PlaneModel.h"
#include "larpandoracontent/LArUtility/RANSAC/RANSAC.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>

using namespace pandora;

namespace lar_content
{

RANSACMethodTool::RANSACMethodTool() :
    m_ransacThreshold(2.5),
    m_ransacIterations(100),
    m_fitIterations(1000),
    m_fitSize(80),
    m_distToEndThreshold(10.0),
    m_distToFitThreshold(6.25),
    m_finishedThreshold(10)
{
}

void RANSACMethodTool::Run(RANSACHitVector &consistentHits, ProtoHitVector &protoHitVector)
{
    if (consistentHits.size() < 3)
        return;

    ParameterVector candidatePoints;
    this->GetCandidatePoints(consistentHits, candidatePoints);

    RANSAC<PlaneModel, 3> estimator(m_ransacThreshold, m_ransacIterations);
    estimator.Estimate(candidatePoints);

    RANSACHitVector primaryResult;
    m_name = "best"; // TODO: Remove;
    const int primaryModelCount(this->RunOverRANSACOutput(estimator, RANSACResult::Best, consistentHits, primaryResult));

    RANSACHitVector secondaryResult;
    m_name = "second"; // TODO: Remove;
    const int secondModelCount(this->RunOverRANSACOutput(estimator, RANSACResult::Second, consistentHits, secondaryResult));

    float primaryFavoured(0);
    float secondaryFavoured(0);

    for (auto hit : primaryResult)
    {
        if (hit.IsFavourable())
            ++primaryFavoured;
    }

    for (auto hit : secondaryResult)
    {
        if (hit.IsFavourable())
            ++secondaryFavoured;
    }

    // INFO: The total is the combination of the RANSAC selected hits, the
    //       considered hits from the fitting, and how many chosen hits are
    //       "favoured". This gives a balance between the model that follows
    //       the most tools, and uses good tools.
    const int primaryTotal(estimator.GetBestInliers().size() + primaryModelCount + primaryFavoured);
    const int secondaryTotal(estimator.GetSecondBestInliers().size() + secondModelCount + secondaryFavoured);

    /**************** Debug **************/
    const int oldPTotal = estimator.GetBestInliers().size() + primaryModelCount;
    const int oldSTotal = estimator.GetSecondBestInliers().size() + secondModelCount;
    const bool oldPrimaryBest = oldPTotal > oldSTotal;
    const bool currentPrimaryBest = primaryTotal > secondaryTotal;
    if (oldPrimaryBest != currentPrimaryBest) {
        std::cout << "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv" << std::endl;
        std::cout << "Old: " << oldPTotal << ", " << oldSTotal << std::endl;
        std::cout << "New: " << primaryTotal << ", " << secondaryTotal << std::endl;
        std::cout << "Fvrd: " << primaryFavoured << ", " << secondaryFavoured << std::endl;
        std::cout << "Sizes: " << primaryResult.size() << ", " << secondaryResult.size() << std::endl;
        std::cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << std::endl;
    }

    ProtoHitVector bestInlyingDebug;
    for (auto hit : primaryResult)
        bestInlyingDebug.push_back(hit.GetProtoHit());
    ProtoHitVector secondBestInlyingDebug;
    for (auto hit : primaryResult)
        secondBestInlyingDebug.push_back(hit.GetProtoHit());
    m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_best", bestInlyingDebug));
    m_allProtoHitsToPlot.push_back(std::make_pair("finalSelectedHits_secondBest", secondBestInlyingDebug));
    /**************** Debug **************/

    const RANSACHitVector bestResult(primaryTotal > secondaryTotal ? primaryResult : secondaryResult);

    for (auto hit : bestResult)
        protoHitVector.push_back(hit.GetProtoHit());
}

//------------------------------------------------------------------------------------------------------------------------------------------

void RANSACMethodTool::GetCandidatePoints(RANSACHitVector &allHits, ParameterVector &candidatePoints)
{
    if (allHits.size() < 1000)
    {
        for (auto hit : allHits)
            candidatePoints.push_back(std::make_shared<RANSACHit>(hit));

        return;
    }

    std::mt19937 eng(allHits.size());
    const unsigned int hitsToUse((allHits.size() - 1) * 0.40);

    // INFO: Fisher-Yates shuffle to get N unique random elements.
    for (unsigned int i = 0; i <= hitsToUse; ++i)
    {
        int j(GetIntsInRange(i, allHits.size() - 1, eng));
        RANSACHit pointI = allHits[i];
        allHits[i] = allHits[j];
        allHits[j] = pointI;

        candidatePoints.push_back(std::make_shared<RANSACHit>(allHits[i]));
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

int RANSACMethodTool::RunOverRANSACOutput(RANSAC<PlaneModel, 3> &ransac, RANSACResult run, RANSACHitVector &hitsToUse, RANSACHitVector &finalHits)
{
    std::map<const CaloHit*, RANSACHit> inlyingHitMap;

    if (run == RANSACResult::Second && !ransac.GetSecondBestModel())
        return 0;

    const ParameterVector currentInliers = run == RANSACResult::Best ? ransac.GetBestInliers() : ransac.GetSecondBestInliers();

    /*****************************************/
    ProtoHitVector currentProtoHitInliers;
    for (auto inlier : currentInliers)
    {
        auto hit = *std::dynamic_pointer_cast<RANSACHit>(inlier);
        currentProtoHitInliers.push_back(hit.GetProtoHit());
    }
    m_allProtoHitsToPlot.push_back(std::make_pair(m_name + "Inliers", currentProtoHitInliers));
    /*****************************************/

    // ATTN: The second model can technically be NULL, so set to first model in that case, as that won't cause any issues.
    //       This is because its possible for the second model to not be populated (if say the first model fits everything),
    //       but this method will not run at all if no models were populated.
    const auto bestModel = ransac.GetBestModel().get();
    const auto secondModel = ransac.GetSecondBestModel() ? ransac.GetSecondBestModel().get() : bestModel;
    const auto currentModel = run == RANSACResult::Best ? bestModel : secondModel;
    const auto otherModel = run == RANSACResult::Best ? secondModel : bestModel;

    RANSACHitVector sortedHits;
    std::list<RANSACHit> nextHits;

    for (auto inlier : currentInliers)
    {
        const auto hit = std::dynamic_pointer_cast<RANSACHit>(inlier);

        if (hit == nullptr)
            throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

        const bool addedHit(RANSACMethodTool::AddToHitMap(*hit, inlyingHitMap));

        if (addedHit)
            sortedHits.push_back(*hit);
    }

    if (sortedHits.size() < 3 || hitsToUse.empty())
    {
        for (auto const& caloProtoPair : inlyingHitMap)
            finalHits.push_back(caloProtoPair.second);

        return 0;
    }

    for (auto hit : hitsToUse)
    {
        if ((inlyingHitMap.count(hit.GetProtoHit().GetParentCaloHit2D()) == 0))
        {
            const float modelDisp(otherModel->ComputeDistanceMeasure(std::make_shared<RANSACHit>(hit)));

            // ATTN: A hit is unfavourable if its from a bad tool, or is in the other model. Unfavourable means it will attempt to not be
            //       used, but can be used if needed.
            const bool isNotInOtherModel(modelDisp > m_ransacThreshold);
            const bool isAlreadyFavourable(hit.IsFavourable());
            nextHits.emplace_back(hit.GetProtoHit(), isNotInOtherModel && isAlreadyFavourable);
        }
    }

    const CartesianVector fitOrigin(currentModel->GetOrigin());
    const CartesianVector fitDirection(currentModel->GetDirection());
    const auto sortByModelDisplacement = [&fitOrigin, &fitDirection] (RANSACHit a, RANSACHit b) {
        float displacementA = (a.GetProtoHit().GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        float displacementB = (b.GetProtoHit().GetPosition3D() - fitOrigin).GetDotProduct(fitDirection);
        return displacementA < displacementB;
    };

    std::sort(sortedHits.begin(), sortedHits.end(), sortByModelDisplacement);
    std::list<RANSACHit> currentPoints3D;

    for (auto hit : sortedHits)
        currentPoints3D.push_back(hit);

    RANSACHitVector hitsToUseForFit;
    RANSACHitVector hitsToAdd;
    RANSACMethodTool::GetHitsForFit(currentPoints3D, hitsToUseForFit, 0, 0);

    int smallIterCount(0);
    int coherentHitCount(0);
    ExtendDirection extendDirection = ExtendDirection::Forward;

    // INFO: For each iteration, try and extend the fit. Fitting is run multiple times per iteration, stopping after the first iteration
    //       that adds hits.
    //       If hits are added, add them to the total hit map and make the new hits available for further iterations of the fit extending.
    //       When no hits remain, repeat from other end of fit, then stop.
    for (unsigned int iter = 0; iter < m_fitIterations; ++iter)
    {
        hitsToAdd.clear();

        for (int fits = 1; fits < 4; ++fits)
        {
            RANSACMethodTool::ExtendFit(nextHits, hitsToUseForFit, hitsToAdd, (m_distToFitThreshold * fits), extendDirection);

            if (hitsToAdd.size() > 0)
                break;
        }

        for (auto hit : hitsToAdd)
        {
            const bool hitAdded = RANSACMethodTool::AddToHitMap(hit, inlyingHitMap);

            if (hitAdded)
                hitsToUseForFit.push_back(hit);

            if (currentPoints3D.empty())
                currentPoints3D.push_back(hit);
        }

        const bool continueFitting = RANSACMethodTool::GetHitsForFit(currentPoints3D, hitsToUseForFit, hitsToAdd.size(), smallIterCount);
        coherentHitCount += hitsToAdd.size();

        // ATTN: If finished first pass, reset and run from opposite end.
        //       If finished second (backwards) pass, stop.
        if (!continueFitting && extendDirection == ExtendDirection::Forward)
        {
            extendDirection = ExtendDirection::Backward;
            hitsToUseForFit.clear();
            currentPoints3D.clear();
            smallIterCount = 0;

            auto it = sortedHits.begin();

            // INFO: Store enough hits for at least 5 fits worth.
            //       This was chosen to ensure the fits had enough hits to be somewhat anchored.
            while (currentPoints3D.size() < (5 * m_fitSize) && it != sortedHits.end())
            {
                currentPoints3D.push_back(*it);
                ++it;
            }

            std::reverse(currentPoints3D.begin(), currentPoints3D.end());
            RANSACMethodTool::GetHitsForFit(currentPoints3D, hitsToUseForFit, 0, 0);
        }
        else if (!continueFitting && extendDirection == ExtendDirection::Backward)
            break;
    }

    for (auto const& caloProtoPair : inlyingHitMap) {
        finalHits.push_back(caloProtoPair.second);
    }

    // ATTN: This count is the count of every considered hit, not the hits that were added.
    //       This allows distinguishing between models that fit to areas with more hits in.
    return coherentHitCount;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool RANSACMethodTool::GetHitsForFit(std::list<RANSACHit> &currentPoints3D, RANSACHitVector &hitsToUseForFit, const int addedHitCount,
    int smallAdditionCount)
{
    // ATTN: Four options:
    //  Added no hits at the end: Stop.
    //  Added small number of hits at end repeatedly: Stop.
    //  Added small number of hits inside fit: Clear and add hits.
    //  Added lots of hits of hits: Don't add hits, just trim vector.
    if (addedHitCount == 0 && currentPoints3D.empty())
        return false;

    if (addedHitCount < 2 && smallAdditionCount > m_finishedThreshold)
        return false;

    if (addedHitCount <= 2 && currentPoints3D.size() != 0)
    {
        unsigned int i(0);
        auto it = currentPoints3D.begin();
        while (i <= m_fitSize && currentPoints3D.size() != 0)
        {
            hitsToUseForFit.push_back(*it);
            it = currentPoints3D.erase(it);

            if (hitsToUseForFit.size() >= m_fitSize)
                hitsToUseForFit.erase(hitsToUseForFit.begin());

            ++i;
        }
    }
    else if (addedHitCount > 0)
    {
        // INFO: Delete all elements past m_fitSize.
        if (hitsToUseForFit.size() > m_fitSize)
        {
            auto it = hitsToUseForFit.begin();
            std::advance(it, m_fitSize);
            hitsToUseForFit.erase(it, hitsToUseForFit.end());
        }
    }

    // ATTN: This keeps track of the number of iterations that add only a small
    // number of hits. Resets if a larger iteration is reached.
    if (addedHitCount < 5 && currentPoints3D.empty())
        ++smallAdditionCount;
    else if (addedHitCount > 15)
        smallAdditionCount = 0;

    return true;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool RANSACMethodTool::AddToHitMap(RANSACHit &hit, std::map<const CaloHit*, RANSACHit> &inlyingHitMap)
{
    const CaloHit* twoDHit =  hit.GetProtoHit().GetParentCaloHit2D();

    if (inlyingHitMap.count(twoDHit) == 0)
    {
        inlyingHitMap.insert({twoDHit, hit});
        return true;
    }
    else
    {
        const float bestMetric(inlyingHitMap.at(twoDHit).GetDisplacement());
        const float metricValue(hit.GetDisplacement());

        if (metricValue < bestMetric)
        {
            inlyingHitMap.insert({twoDHit, hit});
            return true;
        }
    }

    return false;
}

//------------------------------------------------------------------------------------------------------------------------------------------

bool hitIsCloseToEnd(RANSACHit &hit, const CartesianVector &fitStart, const CartesianVector &fitEnd, const CartesianVector &fitDirection, float threshold)
{
    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();
    const float dispFromFitStart((hitPosition - fitStart).GetDotProduct(fitDirection));
    const float dispFromFitEnd((hitPosition - fitEnd).GetDotProduct(fitDirection));

    if (dispFromFitEnd < 0.0 && dispFromFitStart > 0.0)
        return true; // If hit occurs in the middle of the fit, consider it.
    else
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

    const float displacement((hitPosition - projectedPosition).GetCrossProduct(projectedDirection).GetMagnitude());

    hit.SetDisplacement(displacement);
}

void hitDisplacement(RANSACHit &hit, const CartesianVector fitEnd, const CartesianVector fitDirection)
{
    const CartesianVector hitPosition = hit.GetProtoHit().GetPosition3D();
    const float displacement((hitPosition - fitEnd).GetCrossProduct(fitDirection).GetMagnitude());

    hit.SetDisplacement(displacement);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void RANSACMethodTool::ExtendFit(std::list<RANSACHit> &hitsToTestAgainst, RANSACHitVector &hitsToUseForFit, std::vector<RANSACHit> &hitsToAdd,
    const float distanceToFitThreshold, const ExtendDirection extendDirection)
{
    if (hitsToUseForFit.empty())
        return;

    float distanceToEndThreshold(m_distToEndThreshold);

    CartesianPointVector fitPoints;
    for (auto hit : hitsToUseForFit)
        fitPoints.push_back(hit.GetProtoHit().GetPosition3D());

    const unsigned int layerWindow(100);
    const float pitch(LArGeometryHelper::GetWireZPitch(this->GetPandora()));
    const ThreeDSlidingFitResult slidingFit(&fitPoints, layerWindow, pitch);

    CartesianVector fitDirection = slidingFit.GetGlobalMaxLayerDirection();
    CartesianVector fitEnd = slidingFit.GetGlobalMaxLayerPosition();
    CartesianVector fitStart = slidingFit.GetGlobalMinLayerPosition();

    if (extendDirection == ExtendDirection::Backward)
    {
        fitDirection = slidingFit.GetGlobalMinLayerDirection();
        fitEnd = slidingFit.GetGlobalMinLayerPosition();
        fitStart = slidingFit.GetGlobalMaxLayerPosition();
        fitDirection = fitDirection * -1.0;
    }

    // INFO: Check if there is a detector gap, and if there is, extend the fit over it.
    auto distanceToProjectOverGap = LArGeometryHelper::ProjectAcrossGap3D(this->GetPandora(), fitEnd, fitDirection, 2.0, 10);
    if (distanceToProjectOverGap > 0)
        distanceToEndThreshold += std::min(distanceToProjectOverGap, 50.0f);

    // ATTN: This is done in 3 stages to split up the 3 different qualities of hits:
    //          - Hits based on the fit projections.
    //          - Hits based against the fit.
    //          - Use the cached value from the previous two tests, but ignore
    //            the favourable test.
    auto projectedDisplacementTest = [&slidingFit] (RANSACHit &hit, float threshold) {
        projectedHitDisplacement(hit, slidingFit);
        return hit.GetDisplacement() < threshold && hit.IsFavourable();
    };
    auto displacementTest = [&fitEnd, &fitDirection] (RANSACHit &hit, float threshold) {
        hitDisplacement(hit, fitEnd, fitDirection);
        return hit.GetDisplacement() < threshold && hit.IsFavourable();
    };
    auto displacementTestIgnoreFavourableFlag = [] (RANSACHit &hit, float threshold) {
        return hit.GetDisplacement() < threshold;
    };
    std::vector<std::function<bool(RANSACHit&, float)>> tests = {projectedDisplacementTest,
        displacementTest, displacementTestIgnoreFavourableFlag};

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
        newHit.SetPosition3D((*it).GetProtoHit().GetPosition3D(), dispFromFitEnd);
        newHit.SetInterpolated(true);
        hitsComparedInFit.push_back(newHit);
        /*****************************************/

        if (hitIsCloseToEnd((*it), fitStart, fitEnd, fitDirection, distanceToEndThreshold))
        {
            hitsToCheck.push_back(it);
        }
    }

    unsigned int currentTest(0);
    while (hitsToAdd.empty() && currentTest < tests.size())
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

//------------------------------------------------------------------------------------------------------------------------------------------

StatusCode RANSACMethodTool::ReadSettings(const pandora::TiXmlHandle xmlHandle)
{
    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "RANSACThreshold", m_ransacThreshold));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxRANSACIterations", m_ransacIterations));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "MaxFitIterations", m_fitIterations));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "FitImprovementSize", m_fitSize));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "DistanceToProjectFit", m_distToEndThreshold));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "DistanceToProjectOnToFit", m_distToFitThreshold));

    PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadValue(xmlHandle,
        "FinishedThreshold", m_finishedThreshold));

    return STATUS_CODE_SUCCESS;
}

} // namespace lar_content
