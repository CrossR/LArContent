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

void ProjectHitToFit(const CaloHit &twoDHit, const TwoDFitMap &fits, TwoDDisplacementMap &dists)
{
    if (fits.size() != 3)
        throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

    float rL(0.0);
    float rT(0.0);
    fits.at(twoDHit.GetHitType()).GetLocalPosition(twoDHit.GetPositionVector(), rL, rT);

    CartesianVector globalPosition(0.f, 0.f, 0.f);
    fits.at(twoDHit.GetHitType()).GetGlobalPosition(rL, rT, globalPosition);

    dists[twoDHit.GetHitType()].push_back((twoDHit.GetPositionVector() - globalPosition).GetX());
}

//------------------------------------------------------------------------------------------------------------------------------------------

void BuildTwoDFitsForAllViews(const TwoDHitMap &hits, TwoDFitMap &fits, const metricParams &params)
{
    fits.insert({TPC_VIEW_U, TwoDSlidingFitResult(&hits.at(TPC_VIEW_U), params.slidingFitWidth, params.layerPitch)});
    fits.insert({TPC_VIEW_V, TwoDSlidingFitResult(&hits.at(TPC_VIEW_V), params.slidingFitWidth, params.layerPitch)});
    fits.insert({TPC_VIEW_W, TwoDSlidingFitResult(&hits.at(TPC_VIEW_W), params.slidingFitWidth, params.layerPitch)});
}

//------------------------------------------------------------------------------------------------------------------------------------------

void Project3DHitToAllViews(const Pandora &pandora,
        const CartesianVector &hit, TwoDHitMap &hits)
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
    const CartesianPointVector &recoHits, const CaloHitVector &twoDHits,
    threeDMetric &metrics, const metricParams &params,
    const CartesianPointVector &mcHits)
{

    if (recoHits.size() < 10)
        return;

    // Build the initial fit we need.
    const ThreeDSlidingFitResult slidingFit(&recoHits, params.slidingFitWidth, params.layerPitch);
    const ThreeDSlidingFitResult *slidingFitMC = NULL;

    if (mcHits.size() >= 10)
        slidingFitMC = new ThreeDSlidingFitResult(&mcHits, params.slidingFitWidth, params.layerPitch);

    // Setup the variables required for metric calculation.
    const CartesianVector minPosition(slidingFit.GetGlobalMinLayerPosition());
    const CartesianVector maxPosition(slidingFit.GetGlobalMaxLayerPosition());

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

    for (const auto &nextPoint : recoHits)
    {
        try
        {
            const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(nextPoint);

            // Get the position relative to the reco for the point.
            const float rL(slidingFit.GetLongitudinalDisplacement(pointPosition));

            CartesianVector recoPosition(0.f, 0.f, 0.f);
            const StatusCode positionStatusCode(slidingFit.GetGlobalFitPosition(rL, recoPosition));

            if (positionStatusCode != STATUS_CODE_SUCCESS)
                throw StatusCodeException(positionStatusCode);

            if (slidingFitMC != NULL)
            {
                // Get the position relative to the MC for the point.
                const float rLMC(slidingFitMC->GetLongitudinalDisplacement(pointPosition));

                CartesianVector mcTrackPos(0.f, 0.f, 0.f);
                const StatusCode mcPositionStatusCode(slidingFit.GetGlobalFitPosition(rLMC, mcTrackPos));

                if (mcPositionStatusCode != STATUS_CODE_SUCCESS)
                    throw StatusCodeException(mcPositionStatusCode);

                trackDisplacementsSquared.push_back((recoPosition - mcTrackPos).GetMagnitudeSquared());
            }

            // Get the direction relative to the reco for the point.
            CartesianVector direction(0.f, 0.f, 0.f);
            const StatusCode directionStatusCode(slidingFit.GetGlobalFitDirection(rL, direction));

            if (directionStatusCode != STATUS_CODE_SUCCESS)
                throw StatusCodeException(directionStatusCode);

            // Setup the required variables and fill the tree.
            const CartesianVector fitDirection((maxPosition - minPosition).GetUnitVector());

            double dotProduct = fitDirection.GetDotProduct(direction.GetUnitVector());
            dotProduct = acos(dotProduct);

            // If the dot product is greater than 1, or not a number, set to 1.
            if (dotProduct > 1 || dotProduct != dotProduct)
                dotProduct = 1;

            double xDiff = fabs(recoPosition.GetX() - pointPosition.GetX());
            double yDiff = fabs(recoPosition.GetY() - pointPosition.GetY());
            double zDiff = fabs(recoPosition.GetZ() - pointPosition.GetZ());

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
    if (recoHits.size() >= 10 && mcHits.size() >= 10)
    {
        for (const auto &nextPoint : mcHits)
        {
            const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(nextPoint);
            Project3DHitToAllViews(pandora, pointPosition, mcPoints);
        }

        for (const auto &nextPoint : recoHits)
        {
            const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(nextPoint);
            Project3DHitToAllViews(pandora, pointPosition, recoPoints);
        }

        BuildTwoDFitsForAllViews(mcPoints, mcTwoDFits, params);
        BuildTwoDFitsForAllViews(recoPoints, recoTwoDFits, params);

        for (const auto twoDHit : twoDHits)
        {
            ProjectHitToFit(*twoDHit, recoTwoDFits, recoDisplacements);
            ProjectHitToFit(*twoDHit, mcTwoDFits, mcDisplacements);
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
        std::sort(trackDisplacementsSquared.begin(), trackDisplacementsSquared.end());
        std::sort(distancesToFit.begin(), distancesToFit.end());
        std::sort(vectorDifferences.begin(), vectorDifferences.end());

        int element68 = (vectorDifferences.size() * 0.68);

        metrics.recoUDisplacement = recoDisplacements[TPC_VIEW_U];
        metrics.recoVDisplacement = recoDisplacements[TPC_VIEW_V];
        metrics.recoWDisplacement = recoDisplacements[TPC_VIEW_W];

        metrics.acosDotProductAverage = vectorDifferences[element68];
        metrics.distanceToFitAverage = distancesToFit[element68];
        metrics.lengthOfTrack = (maxPosition - minPosition).GetMagnitude();

        metrics.numberOf3DHits = recoHits.size(); // Regardless of errors, this is the number of hits we were given.
        metrics.valuesHaveBeenSet = errorCases::SUCCESSFULLY_SET;

        if (slidingFitMC != NULL)
        {
            metrics.trackDisplacementAverageMC = trackDisplacementsSquared[element68];

            metrics.mcUDisplacement = mcDisplacements[TPC_VIEW_U];
            metrics.mcVDisplacement = mcDisplacements[TPC_VIEW_V];
            metrics.mcWDisplacement = mcDisplacements[TPC_VIEW_W];
        }
    }
}
}
