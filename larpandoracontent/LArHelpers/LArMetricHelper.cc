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

#include "larpandoracontent/LArHelpers/LArObjectHelper.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

using namespace pandora;

namespace lar_content
{

void LArMetricHelper::GetThreeDMetrics(const CartesianPointVector *const hits, const ThreeDSlidingFitResult *const slidingFit,
                                       threeDMetric& metrics, const ThreeDSlidingFitResult *const slidingFitMC)
{
    // Setup the variables required for metric calculation.
    const CartesianVector minPosition(slidingFit->GetGlobalMinLayerPosition());
    const CartesianVector maxPosition(slidingFit->GetGlobalMaxLayerPosition());

    std::vector<double> vectorDifferences;
    std::vector<double> distancesToFit;
    std::vector<double> trackDisplacementsSquared;
    metrics.numberOfErrors = 0;

    for (const auto &nextPoint : *hits)
    {
        try {
            const CartesianVector pointPosition = LArObjectHelper::TypeAdaptor::GetPosition(nextPoint);

            // Get the position relative to the reco for the point.
            const float rL(slidingFit->GetLongitudinalDisplacement(pointPosition));

            CartesianVector recoPosition(0.f, 0.f, 0.f);
            const StatusCode positionStatusCode(slidingFit->GetGlobalFitPosition(rL, recoPosition));

            if (positionStatusCode != STATUS_CODE_SUCCESS)
                throw StatusCodeException(positionStatusCode);

            if (slidingFitMC != NULL)
            {
                // Get the position relative to the MC for the point.
                const float rLMC(slidingFitMC->GetLongitudinalDisplacement(pointPosition));

                CartesianVector mcTrackPos(0.f, 0.f, 0.f);
                const StatusCode mcPositionStatusCode(slidingFitMC->GetGlobalFitPosition(rLMC, mcTrackPos));

                if (mcPositionStatusCode != STATUS_CODE_SUCCESS)
                    throw StatusCodeException(mcPositionStatusCode);

                trackDisplacementsSquared.push_back((recoPosition - mcTrackPos).GetMagnitudeSquared());
            }

            // Get the direction relative to the reco for the point.
            CartesianVector direction(0.f, 0.f, 0.f);
            const StatusCode directionStatusCode(slidingFit->GetGlobalFitDirection(rL, direction));

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
        } catch (const StatusCodeException &statusCodeException1) {

            // TODO: Check this over.
            // Currently, if this is set for every single hit, its thrown
            // away at the end.  Is that suitable?
            //
            // I.e. we error on every hit, so the error case is hit, and
            // this is set to -999 and is ignored from the metrics. Does it
            // make sense to keep this? Or not, since it will not
            // contribute to any other errors when it is a track that didn't
            // actually add anything to the 3D reco.
            metrics.numberOfErrors++;

            if (statusCodeException1.GetStatusCode() == STATUS_CODE_FAILURE)
                throw statusCodeException1;
        }
    }

    // If there is nothing to log, make sure the metric is set to
    // indicate this. Then the default values will be filled in instead.
    if (distancesToFit.size() == 0) {
        metrics.valuesHaveBeenSet = errorCases::TRACK_BUILDING_ERROR;
    } else {
        // Sort all the vectors and get the 68% element to log out.
        std::sort(trackDisplacementsSquared.begin(), trackDisplacementsSquared.end());
        std::sort(distancesToFit.begin(), distancesToFit.end());
        std::sort(vectorDifferences.begin(), vectorDifferences.end());
        int element68 = (vectorDifferences.size() * 0.68);

        metrics.acosDotProductAverage = vectorDifferences[element68];
        metrics.distanceToFitAverage = distancesToFit[element68];
        metrics.lengthOfTrack = (maxPosition - minPosition).GetMagnitude();

        metrics.numberOf3DHits = hits->size(); // Regardless of errors, this is the number of hits we were given.
        metrics.valuesHaveBeenSet = errorCases::SUCCESSFULLY_SET;

        if (slidingFitMC != NULL)
            metrics.trackDisplacementAverageMC = trackDisplacementsSquared[element68];
    }
}
}
