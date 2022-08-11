/**
 *  @file   larpandoracontent/LArCheating/CheatingHitTrackShowerIdAlgorithm.cc
 *
 *  @brief  Header file for the cheated track shower id algorithm.
 *
 *  $Log: $
 */
#ifndef LAR_CHEATING_HIT_TRACK_SHOWER_ID_ALGORITHM_H
#define LAR_CHEATING_HIT_TRACK_SHOWER_ID_ALGORITHM_H 1

#include "Pandora/Algorithm.h"

namespace lar_content
{

/**
 *  @brief  CheatingHitTrackShowerIdAlgorithm class
 */
class CheatingHitTrackShowerIdAlgorithm : public pandora::Algorithm
{
public:

    CheatingHitTrackShowerIdAlgorithm();

private:
    pandora::StatusCode Run();
    pandora::StatusCode ReadSettings(const pandora::TiXmlHandle xmlHandle);

    pandora::StringVector m_caloHitListNames; ///< Name of input calo hit list
};

} // namespace lar_dl_content

#endif // LAR_CHEATING_HIT_TRACK_SHOWER_ID_ALGORITHM_H
