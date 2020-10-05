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
#include "larpandoracontent/LArUtility/RANSAC/RANSAC.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/HitCreationBaseTool.h"
#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

#include <Eigen/Core>

namespace lar_content
{

class RANSACHit : public AbstractParameter
{
public:

    typedef ThreeDHitCreationAlgorithm::ProtoHit ProtoHit;
    typedef ThreeDHitCreationAlgorithm::ProtoHitVector ProtoHitVector;

    /**
     *  @brief  Constructor.
     *
     *  @param  protoHit    The base ProtoHit.
     *  @param  favourable  If the hit is favoured or not.
     */
    RANSACHit(const ProtoHit &protoHit, const bool favourable);

    /**
     *  @brief  Get if the proto hit is favourable or not.
     *
     *  @return boolean
     */
    bool IsFavourable() const;

    /**
     *  @brief  Get the associated ProtoHit.
     *
     *  @return ProtoHit
     */
    ProtoHit GetProtoHit() const;

    /**
     *  @brief  Get the stored displacement.
     *
     *  @return float
     */
    float GetDisplacement() const;

    /**
     *  @brief  Set the displacement of this hit, relative to the current fit.
     *          Only set if the displacement is lower than the current. This
     *          allows many values to be given, but only the best stored.
     *
     *  @param  displacement  The current displacement value;
     */
    void SetDisplacement(float displacement);

    /**
     *  @brief  Get the 3D position of the protoHit as an Eigen::Vector3f.
     *          This is to make Eigen calculations easier in the RANSAC methods.
     *
     *  @return Eigen::Vector3f
     */
    Eigen::Vector3f GetVector() const;

    float& operator[](int i);

private:
    ProtoHit           m_protoHit;         ///< The parent protoHit.
    bool               m_favourable;       ///< Whether the hit is favourable or not.
    float              m_displacement;     ///< The displacement of this hit from the fit
};


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

    enum RANSACResult {
        Best,
        Second
    };

    typedef std::vector<RANSACHit> RANSACHitVector;

    LArRANSACMethod(float pitch, RANSACHitVector &consistentHits, const pandora::Pandora *const pandora);
    void Run(ProtoHitVector &protoHitVector);

    std::vector<std::pair<std::string, ProtoHitVector>> m_allProtoHitsToPlot;
    int m_iter;
    std::string m_name;

private:

    RANSACHitVector m_consistentHits;
    const float m_pitch;
    const pandora::Pandora *const m_pandora;

    /**
     *  @brief  Given a RANSAC model, run over it and produce the full 3D model
     *  by extending and improving the initial model.
     *
     *  @param  ransac          The complete RANSAC object, containing both the models.
     *  @param  run             Enum describing which run to consider (Best or Second).
     *  @param  hitsToUse       The hits to be used in the fit extending.
     *  @param  protoHitVector  Vector to store the final complete model in.
     */
    int RunOverRANSACOutput(RANSAC<PlaneModel, 3> &ransac, RANSACResult run, RANSACHitVector &hitsToUse, RANSACHitVector &finalHits);


    /**
     *  @brief  Given a set of selected hits and candidate hits, try and add
     *  candidate hits using a sliding fit.
     *
     *  @param  hitsToTestAgainst       The list of hits that should be considered i.e. hits which could be added.
     *  @param  hitsToUseForFit         The hits that should be used for the current sliding fit.
     *  @param  hitsToAdd               Vector to store any hits that are close enough to the current fit to be used.
     *  @param  distanceToFitThreshold  Threshold for how far a hit can be from the fit (perpendicular distance).
     *  @param  extendDirection         Enum describing the current mode of running (forward or backwards).
     */
     void ExtendFit(std::list<RANSACHit> &hitsToTestAgainst, RANSACHitVector &hitsToUseForFit,
             std::vector<RANSACHit> &hitsToAdd, const float distanceToFitThreshold,
             const ExtendDirection extendDirection);

    /**
     *  @brief  Given a candidate hit, check it and see if it is worth storing
     *          as the best 3D hit. This is done by comparing the displacement
     *          score.
     *
     *  @param  hit  The RANSACHit to consider adding to the hit map.
     *  @param  inlyingHitMap  The hit map, matching a calo hit to its best RANSACHit.
     */
     bool AddToHitMap(RANSACHit &hit, std::map<const pandora::CaloHit*, RANSACHit> &inlyingHitMap);

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
     *  @param  currentPoints3D      The list of hits to sample from.
     *  @param  hitsToAdd            The vector of hits to populated to use for fitting, if appropriate.
     *  @param  addedHitCount        The number of hits added in the last fit.
     *  @param  smallAdditionCount   The current number of iterations that added a small number of hits.
     */
     bool GetHitsForFit(std::list<RANSACHit> &currentPoints3D, RANSACHitVector &hitsToAdd,
             const int addedHitCount, int smallAdditionCount);

    /**
     *  @brief  Get the candidate hits to use for the RANSAC method. This will
     *          return all hits for small vectors, and randomly sample 40% of
     *          the hits for larger vectors (>1000 hits). Uses the Fisher-Yates
     *          shuffle to get unique elements.
     *
     *  @param  allHits  Vector of all candidate hits.
     *  @param  candidatePoints  Vector to store the chosen candidate hits in.
     */
    void GetCandidatePoints(RANSACHitVector &allHits, ParameterVector &candidatePoints);
};

//------------------------------------------------------------------------------------------------------------------------------------------

inline LArRANSACMethod::LArRANSACMethod(float pitch, RANSACHitVector &consistentHits, const pandora::Pandora *const pandora) :
    m_consistentHits(consistentHits),
    m_pitch(pitch),
    m_pandora(pandora)
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline RANSACHit::RANSACHit(const ProtoHit &protoHit, const bool favoured) :
    m_protoHit(protoHit),
    m_favourable(favoured),
    m_displacement(std::numeric_limits<float>::max())
{
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline bool RANSACHit::IsFavourable() const
{
    return m_favourable;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline ThreeDHitCreationAlgorithm::ProtoHit RANSACHit::GetProtoHit() const
{
    return m_protoHit;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline float RANSACHit::GetDisplacement() const
{
    return m_displacement;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline void RANSACHit::SetDisplacement(float displacement)
{
    if (displacement < m_displacement)
        m_displacement = displacement;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline Eigen::Vector3f RANSACHit::GetVector() const
{
    Eigen::Vector3f position;
    position(0) = m_protoHit.GetPosition3D().GetX();
    position(1) = m_protoHit.GetPosition3D().GetY();
    position(2) = m_protoHit.GetPosition3D().GetZ();

    return position;
}

//------------------------------------------------------------------------------------------------------------------------------------------

inline float& RANSACHit::operator[](int i)
{
    if(i < 3)
        return this->GetVector()(i);

    throw std::runtime_error("Index exceeded bounds.");
}

} // namespace lar_content

#endif // #ifndef LAR_RANSAC_FIT_EXTEND
