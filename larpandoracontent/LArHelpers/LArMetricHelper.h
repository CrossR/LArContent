/**
 *  @file   larpandoracontent/LArHelpers/LArMetricHelper.h
 *
 *  @brief  Header file for the metric generation helper class.
 *
 *  $Log: $
 */
#ifndef LAR_METRIC_HELPER_H
#define LAR_METRIC_HELPER_H 1

#include "larpandoracontent/LArObjects/LArThreeDSlidingFitResult.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"

namespace lar_content
{

enum errorCases {
    NOT_SET,
    ERROR,
    SUCCESSFULLY_SET,
    NON_NEUTRINO,
    NON_FINAL_STATE,
    NON_TRACK,
    TRACK_BUILDING_ERROR,
    NO_VERTEX_ERROR
};

// Struct for storing metric result.
// This lets me move the code to somewhere more useful.
struct threeDMetric {
    errorCases valuesHaveBeenSet;
    double acosDotProductAverage;
    double trackDisplacementAverageMC;
    double distanceToFitAverage;
    double distanceToFitAverageYZ;
    double numberOfErrors;
    double lengthOfTrack;
    double numberOf3DHits;
};

/**
 *  @brief  LArMetricHelper class
 */
class LArMetricHelper
{
public:
    /**
     *  @brief  Generate metrics for generated 3D hits.
     *
     *  @param  hits A vector (CartesianPointVector) of hits to generate metrics for.
     *  @param  slidingFit A sliding fit that is used to base the data driven metrics on.
     *  @param  metrics A struct to contain the metric results.
     *  @param  slidingFitMC A second sliding fit for MC driven metrics (optional).
     */
    static void GetThreeDMetrics(const pandora::CartesianPointVector *const hits,const ThreeDSlidingFitResult *const slidingFit,
            threeDMetric& metrics, const ThreeDSlidingFitResult *const slidingFitMC);
};

} // namespace lar_content

#endif // #ifndef LAR_METRIC_HELPER_H
