/**
 *  @file   larpandoracontent/LArHelpers/LArMetricHelper.cc
 *
 *  @brief  Implementation of the metric helper class.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArHelpers/LArMetricHelper.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"
#include "larpandoracontent/LArObjects/LArTwoDSlidingFitResult.h"

#include "larpandoracontent/LArHelpers/LArObjectHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

using namespace pandora;

namespace lar_content
{

//------------------------------------------------------------------------------------------------------------------------------------------

float GetAverageDisplacement(std::vector<double> &displacements)
{
    if (displacements.size() == 0)
        return -999.0;

    std::sort(displacements.begin(), displacements.end());
    int element68 = (displacements.size() * 0.68);
    return displacements[element68];
}

//------------------------------------------------------------------------------------------------------------------------------------------

void ProjectHitToFit(const CaloHit &twoDHit, const TwoDFitMap &fits, CartesianVector &globalPosition)
{
    if (fits.size() != 3)
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

    float rL(0.0);
    float rT(0.0);
    fits.at(twoDHit.GetHitType()).GetLocalPosition(twoDHit.GetPositionVector(), rL, rT);

    fits.at(twoDHit.GetHitType()).GetGlobalFitPosition(rL, globalPosition);
}

//------------------------------------------------------------------------------------------------------------------------------------------

void BuildTwoDFitsForAllViews(const TwoDHitMap &hits, TwoDFitMap &fits, const metricParams &params)
{
    fits.insert({TPC_VIEW_U, TwoDSlidingFitResult(&hits.at(TPC_VIEW_U), params.slidingFitWidth, params.layerPitch)});
    fits.insert({TPC_VIEW_V, TwoDSlidingFitResult(&hits.at(TPC_VIEW_V), params.slidingFitWidth, params.layerPitch)});
    fits.insert({TPC_VIEW_W, TwoDSlidingFitResult(&hits.at(TPC_VIEW_W), params.slidingFitWidth, params.layerPitch)});
}

//------------------------------------------------------------------------------------------------------------------------------------------

void Project3DHitToAllViews(const Pandora &pandora, const CartesianVector &hit, TwoDHitMap &hits)
{
    if (hits.size() == 0)
    {
        hits.insert({TPC_VIEW_U, {LArGeometryHelper::ProjectPosition(pandora, hit, TPC_VIEW_U)}});
        hits.insert({TPC_VIEW_V, {LArGeometryHelper::ProjectPosition(pandora, hit, TPC_VIEW_V)}});
        hits.insert({TPC_VIEW_W, {LArGeometryHelper::ProjectPosition(pandora, hit, TPC_VIEW_W)}});
    }
    else
    {
        hits.at(TPC_VIEW_U).push_back(LArGeometryHelper::ProjectPosition(pandora, hit, TPC_VIEW_U));
        hits.at(TPC_VIEW_V).push_back(LArGeometryHelper::ProjectPosition(pandora, hit, TPC_VIEW_V));
        hits.at(TPC_VIEW_W).push_back(LArGeometryHelper::ProjectPosition(pandora, hit, TPC_VIEW_W));
    }
}

//------------------------------------------------------------------------------------------------------------------------------------------

void LArMetricHelper::GetThreeDMetrics(const Pandora &pandora,
    const ParticleFlowObject *const pPfo,
    const CartesianPointVector &recoHits, const CaloHitVector &twoDHits,
    threeDMetric &metrics, const metricParams &params,
    const CartesianPointVector &mcHits)
{

    // Build the initial fit we need.
    const ThreeDSlidingFitResult *slidingFit = NULL;
    const ThreeDSlidingFitResult *slidingFitMC = NULL;

    try
    {
        slidingFit = new ThreeDSlidingFitResult(&recoHits, params.slidingFitWidth, params.layerPitch);
    }
    catch (const StatusCodeException &statusCodeException1)
    {
        slidingFit = NULL;
    }

    try
    {
        slidingFitMC = new ThreeDSlidingFitResult(&mcHits, params.slidingFitWidth, params.layerPitch);
    }
    catch (const StatusCodeException &statusCodeException1)
    {
        slidingFitMC = NULL;
    }

    // Make maps to store 2D fits, as well as the hits used to build them, and
    // the displacements from the fits.
    TwoDHitMap recoPoints;
    TwoDHitMap mcPoints;

    TwoDFitMap recoTwoDFits;
    TwoDFitMap mcTwoDFits;

    TwoDDisplacementMap recoDisplacements;
    TwoDDisplacementMap mcDisplacements;

    std::vector<double> vectorDifferences;
    std::vector<double> distancesToFit;
    std::vector<double> trackDisplacementsSquared;
    metrics.numberOfErrors = 0;
    int numberOfMCErrors = 0;

    for (const auto nextPoint : recoHits)
    {
        if (slidingFit == NULL)
            continue;

        try
        {
            // Get the position relative to the reco for the point.
            const float rL(slidingFit->GetLongitudinalDisplacement(nextPoint));

            CartesianVector recoPosition(0.f, 0.f, 0.f);
            const StatusCode positionStatusCode(slidingFit->GetGlobalFitPosition(rL, recoPosition));

            if (positionStatusCode != STATUS_CODE_SUCCESS)
                throw StatusCodeException(positionStatusCode);

            try
            {
                if (slidingFitMC != NULL)
                {
                    // Get the position relative to the MC for the point.
                    const float rLMC(slidingFitMC->GetLongitudinalDisplacement(nextPoint));

                    CartesianVector mcTrackPos(0.f, 0.f, 0.f);
                    const StatusCode mcPositionStatusCode(slidingFitMC->GetGlobalFitPosition(rLMC, mcTrackPos));

                    if (mcPositionStatusCode != STATUS_CODE_SUCCESS)
                        throw StatusCodeException(mcPositionStatusCode);

                    trackDisplacementsSquared.push_back((recoPosition - mcTrackPos).GetMagnitudeSquared());
                }
            }
            catch (const StatusCodeException&)
            {
                numberOfMCErrors++;
            }

            // Get the direction relative to the reco for the point.
            CartesianVector direction(0.f, 0.f, 0.f);
            const StatusCode directionStatusCode(slidingFit->GetGlobalFitDirection(rL, direction));

            if (directionStatusCode != STATUS_CODE_SUCCESS)
                throw StatusCodeException(directionStatusCode);

            // Setup the required variables and fill the tree.
            const CartesianVector minPosition(slidingFit->GetGlobalMinLayerPosition());
            const CartesianVector maxPosition(slidingFit->GetGlobalMaxLayerPosition());
            const CartesianVector fitDirection((maxPosition - minPosition).GetUnitVector());

            double dotProduct = fitDirection.GetDotProduct(direction.GetUnitVector());
            dotProduct = acos(dotProduct);

            // If the dot product is greater than 1, or not a number, set to 1.
            if (dotProduct > 1 || dotProduct != dotProduct)
                dotProduct = 1;

            double xDiff = fabs(recoPosition.GetX() - nextPoint.GetX());
            double yDiff = fabs(recoPosition.GetY() - nextPoint.GetY());
            double zDiff = fabs(recoPosition.GetZ() - nextPoint.GetZ());

            double combinedDiff = sqrt(1.0/3.0 * (pow(xDiff, 2) + pow(yDiff, 2) + pow(zDiff, 2)));

            vectorDifferences.push_back(dotProduct);
            distancesToFit.push_back(combinedDiff);
        }
        catch (const StatusCodeException &statusCodeException1)
        {
            metrics.numberOfErrors++;

            if (statusCodeException1.GetStatusCode() == STATUS_CODE_FAILURE)
                throw statusCodeException1;
        }
    }

    // Only run when possible for both
    if (slidingFit != NULL)
    {
        for (const auto &nextPoint : recoHits)
        {
            Project3DHitToAllViews(pandora, nextPoint, recoPoints);
        }

        BuildTwoDFitsForAllViews(recoPoints, recoTwoDFits, params);

        if (slidingFitMC != NULL)
        {
            for (const auto &nextPoint : mcHits)
            {
                Project3DHitToAllViews(pandora, nextPoint, mcPoints);
            }

            BuildTwoDFitsForAllViews(mcPoints, mcTwoDFits, params);
        }

        for (const auto twoDHit : twoDHits)
        {
            CartesianVector recoFitPosition(0.f, 0.f, 0.f);

            ProjectHitToFit(*twoDHit, recoTwoDFits, recoFitPosition);

            recoDisplacements[twoDHit->GetHitType()].push_back((twoDHit->GetPositionVector() - recoFitPosition).GetMagnitude());

            if (slidingFitMC != NULL)
            {
                CartesianVector mcFitPosition(0.f, 0.f, 0.f);
                ProjectHitToFit(*twoDHit, mcTwoDFits, mcFitPosition);
                mcDisplacements[twoDHit->GetHitType()].push_back((recoFitPosition - mcFitPosition).GetMagnitude());
            }
        }
    }

    // If there is nothing to log, make sure the metric is set to
    // indicate this. Then the default values will be filled in instead.
    if (distancesToFit.size() == 0)
    {
        metrics.valuesHaveBeenSet = errorCases::TRACK_BUILDING_ERROR;
    }
    else
    {
        // Sort all the vectors and get the 68% element to log out.
        std::sort(distancesToFit.begin(), distancesToFit.end());
        std::sort(vectorDifferences.begin(), vectorDifferences.end());

        int element68 = (vectorDifferences.size() * 0.68);

        metrics.recoUDisplacement = GetAverageDisplacement(recoDisplacements[TPC_VIEW_U]);
        metrics.recoVDisplacement = GetAverageDisplacement(recoDisplacements[TPC_VIEW_V]);
        metrics.recoWDisplacement = GetAverageDisplacement(recoDisplacements[TPC_VIEW_W]);

        metrics.acosDotProductAverage = vectorDifferences[element68];
        metrics.distanceToFitAverage = distancesToFit[element68];

        const CartesianVector minPosition(slidingFit->GetGlobalMinLayerPosition());
        const CartesianVector maxPosition(slidingFit->GetGlobalMaxLayerPosition());
        metrics.lengthOfTrack = (maxPosition - minPosition).GetMagnitude();

        metrics.valuesHaveBeenSet = errorCases::SUCCESSFULLY_SET;

        if (slidingFitMC != NULL)
        {
            if (trackDisplacementsSquared.size() > 0)
            {
                std::sort(trackDisplacementsSquared.begin(), trackDisplacementsSquared.end());
                int mcElement68 = (trackDisplacementsSquared.size() * 0.68);
                metrics.trackDisplacementAverageMC = trackDisplacementsSquared[mcElement68];
            }

            metrics.mcUDisplacement = GetAverageDisplacement(mcDisplacements[TPC_VIEW_U]);
            metrics.mcVDisplacement = GetAverageDisplacement(mcDisplacements[TPC_VIEW_V]);
            metrics.mcWDisplacement = GetAverageDisplacement(mcDisplacements[TPC_VIEW_W]);
        }
    }

    ClusterList clusterList;
    LArPfoHelper::GetTwoDClusterList(pPfo, clusterList);
    int totalNumberOf2DHits = 0;

    for (auto cluster : clusterList)
        totalNumberOf2DHits += cluster->GetNCaloHits();

    metrics.numberOf3DHits = recoHits.size();
    metrics.numberOf2DHits = totalNumberOf2DHits;
}
}
