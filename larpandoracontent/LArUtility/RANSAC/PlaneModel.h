/**
 *  @file   larpandoracontent/LArUtility/RANSAC/PlaneModel.h
 *
 *  @brief  Header file for the PlaneModel, to be used in RANSAC.
 *
 *  $Log: $
 */
#ifndef LAR_PLANE_MODEL_RANSAC_H
#define LAR_PLANE_MODEL_RANSAC_H 1

#include "larpandoracontent/LArUtility/RANSAC/AbstractModel.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

#include <Eigen/Core>
#include <Eigen/SVD>

namespace lar_content
{

/**
 *  @brief  Class that implements a 3D Point in the form RANSAC needs.
 *          We spin the position out into an Eigen::Vector3f, to aid
 *          building of a matrix and more later.
 */
class Point3D : public AbstractParameter
{
public:

    Point3D(ThreeDHitCreationAlgorithm::ProtoHit &p);

    ThreeDHitCreationAlgorithm::ProtoHit m_ProtoHit;
    Eigen::Vector3f m_Point3D;

    float& operator[](int i);
};

/**
 *  @brief  Class that implements a PlaneModel, to be fit using RANSAC.
 */
class PlaneModel: public AbstractModel<3>
{
private:

    Eigen::Vector3f m_direction;
    Eigen::Vector3f m_origin;

public:

    /**
     *  @brief  Project point on to the current line and work out the distance
     *  from the line.
     *
     *  @param param  The Point3D to compare to the current line.
     */
    virtual double ComputeDistanceMeasure(SharedParameter param) override;

    PlaneModel(ParameterVector inputParams) { Initialize(inputParams); };
    virtual ~PlaneModel() {};

    pandora::CartesianVector GetDirection();

    pandora::CartesianVector GetOrigin();

    virtual void Initialize(const ParameterVector &inputParams) override;
    virtual std::pair<double, ParameterVector> Evaluate(const ParameterVector &paramsToEval, double threshold) override;
};

} // namespace lar_content

#endif // LAR_PLANE_MODEL_RANSAC_H
