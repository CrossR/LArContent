/**
 *  @file   larpandoracontent/LArThreeDReco/LArHitCreation/RANSACMethod.h
 *
 *  @brief  Header file for the RANSAC related methods.
 *
 *  $Log: $
 */
#ifndef LAR_RANSAC_FIT_EXTEND
#define LAR_RANSAC_FIT_EXTEND 1

#include "larpandoracontent/LArUtility/RANSAC/AbstractModel.h"
#include "larpandoracontent/LArUtility/RANSAC/PlaneModel.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/HitCreationBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

namespace lar_content
{

/**
 *  @brief  LArRANSACMethod class
 */
class LArRANSACMethod
{
public:

    typedef ThreeDHitCreationAlgorithm::ProtoHit ProtoHit;
    typedef ThreeDHitCreationAlgorithm::ProtoHitVector ProtoHitVector;

    enum ExtendDirection {
        Forward,
        Backward
    };

    LArRANSACMethod(float pitch, ProtoHitVector &consistentHits);
    void Run(ProtoHitVector &protoHitVector);

    std::vector<std::pair<std::string, ProtoHitVector>> m_allProtoHitsToPlot;
    std::vector<std::pair<std::string, ParameterVector>> m_parameterVectors;

private:

    ProtoHitVector m_consistentHits;
    const float m_pitch;

    /**
     *  @brief  TODO
     *
     *  @param  TODO
     */
    int RunOverRANSACOutput(PlaneModel &currentModel, ParameterVector &currentInliers, ProtoHitVector &hitsToUse, ProtoHitVector &protoHitVector,
        std::string name
    );


    /**
     *  @brief  Given a set of selected hits and candidate hits, try and add candidate hits using a sliding fit.
     *
     *  @param  TODO
     */
     void ExtendFit(ProtoHitVector &hitsToTestAgainst, ProtoHitVector &hitsToUseForFit,
             std::vector<std::pair<ProtoHit, float>> &hitsAddedToFit, const float distanceToFitThreshold,
             const ExtendDirection extendDirection,
             int iter, std::string name);

    /**
     *  @brief  Given a candidate hit, check it and see if it is worth storing
     *          as the best 3D hit. This is done by comparing the displacement
     *          score.
     *
     *  @param  hit  The ProtoHit to consider adding to the hit map.
     *  @param  inlyingHitMap  The hit map, matching a calo hit to its best 3D hit and the score of that hit.
     *  @param  displacement  The displacement/score of the candidate hit. Used to check if the current hit is better.
     */
     bool AddToHitMap(ProtoHit hit, std::map<const pandora::CaloHit*, std::pair<ProtoHit, float>> &inlyingHitMap,
             float displacement);

    /**
     *  @brief  Get the hits that should be used for the next fit, assuming a
     *          fit is needed. Returns a boolean if the next fit should be run
     *          or not.
     *
     *          This is decided using a few things:
     *              - If the fit has reached the end and there is no more hits,
     *                stop.
     *              - If the fitting has only been adding a small number of
     *                hits for too long, stop.
     *              - If the fitting only added a small number of hits, but we
     *                still have hits, clear and move to the next N hits.
     *              - If the fitting added lots of hits, trim to the right size
     *                only, such that the fit can continue extending.
     *
     *  @param  currentPoints3D   The list of hits to sample from.
     *  @param  hitsToUseForFit   The vector of hits to populated to use for fitting, if appropriate.
     *  @param  addedHitCount     The number of hits added in the last fit.
     *  @param  hitsToUseForFit   The current number of iterations that added a small number of hits.
     */
     bool GetHitsForFit(std::list<ProtoHit> &currentPoints3D, ProtoHitVector &hitsToUseForFit,
             const int addedHitCount, int smallAdditionCount);
};

//------------------------------------------------------------------------------------------------------------------------------------------

inline LArRANSACMethod::LArRANSACMethod(float pitch, ProtoHitVector &consistentHits) :
    m_consistentHits(consistentHits),
    m_pitch(pitch)
{
}

} // namespace lar_content

#endif // #ifndef LAR_RANSAC_FIT_EXTEND
