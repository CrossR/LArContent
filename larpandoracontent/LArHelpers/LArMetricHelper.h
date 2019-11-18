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
    double numberOfErrors;
    double lengthOfTrack;
    double numberOf3DHits;

    std::vector<float> recoUDisplacement;
    std::vector<float> recoVDisplacement;
    std::vector<float> recoWDisplacement;

    std::vector<float> mcUDisplacement;
    std::vector<float> mcVDisplacement;
    std::vector<float> mcWDisplacement;
};

// Struct for required parameters for the metric generation.
struct metricParams {
    float layerPitch;
    float slidingFitWidth;
};

typedef std::map<pandora::HitType, pandora::CartesianPointVector> TwoDHitMap;
typedef std::map<pandora::HitType, TwoDSlidingFitResult> TwoDFitMap;
typedef std::map<pandora::HitType, std::vector<float>> TwoDDisplacementMap;

/**
 *  @brief  LArMetricHelper class
 */
class LArMetricHelper
{
public:
    /**
     *  @brief  Generate metrics for generated 3D hits.
     *
     *  @param  pandora The current pandora instance, to interface with the geometry helper.
     *  @param  hits A vector (CartesianPointVector) of hits to generate metrics for.
     *  @param  twoDhits A vector (CaloHitVector) of the parent calo hits.
     *  @param  params A struct that contains required parameters for the metric generation.
     *  @param  metrics A struct to contain the metric results.
     *  @param  mcHits A second set of MC 3D hits for MC driven metrics (optional).
     */
    static void GetThreeDMetrics(const pandora::Pandora &pandora,
            const pandora::CartesianPointVector &recoHits, const pandora::CaloHitVector &twoDHits,
            threeDMetric &metrics, const metricParams&params,
            const pandora::CartesianPointVector &mcHits = {});
};

} // namespace lar_content

#endif // #ifndef LAR_METRIC_HELPER_H
