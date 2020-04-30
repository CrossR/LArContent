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
     *  @brief  Given a set of selected hits and candidate hits, try and add candidate hits using a sliding fit.
     *
     *  @param  TODO
     */
     bool AddToHitMap(ProtoHit hit, std::map<const pandora::CaloHit*, std::pair<ProtoHit, float>> &inlyingHitMap,
             float displacement);

    /**
     *  @brief  TODO
     *
     *  @param  TODO
     *  @param  TODO Evaluate public/private of the various new methods.
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
