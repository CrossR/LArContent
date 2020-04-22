/**
 *  @file   larpandoracontent/LArUtility/RANSAC/PlaneModel.cc
 *
 *  @brief  Implementation file for the PlaneModel, to be used in RANSAC.
 *
 *  $Log: $
 */

#include "larpandoracontent/LArUtility/RANSAC/AbstractModel.h"
#include "larpandoracontent/LArUtility/RANSAC/PlaneModel.h"

#include "larpandoracontent/LArThreeDReco/LArHitCreation/ThreeDHitCreationAlgorithm.h"

#include <Eigen/Core>
#include <Eigen/SVD>

namespace lar_content
{

//------------------------------------------------------------------------------------------------------------------------------------------

Point3D::Point3D(ProtoHit &p)
{
    m_ProtoHit = p;
    m_Point3D(0) = p.GetPosition3D().GetX();
    m_Point3D(1) = p.GetPosition3D().GetY();
    m_Point3D(2) = p.GetPosition3D().GetZ();
}

//------------------------------------------------------------------------------------------------------------------------------------------

float& Point3D::operator[](int i)
{
    if(i < 3)
        return m_Point3D(i);

    throw std::runtime_error("Point3D::Operator[] - Index exceeded bounds.");
}

//------------------------------------------------------------------------------------------------------------------------------------------

virtual double PlaneModel::ComputeDistanceMeasure(SharedParameter param) override
{
    auto currentPoint = std::dynamic_pointer_cast<Point3D>(param);
    if(currentPoint == nullptr)
        throw std::runtime_error("PlaneModel::ComputeDistanceMeasure() - Passed parameter are not of type Point3D.");

    auto point = *currentPoint;
    auto currentPos = point.m_Point3D - m_origin;

    Eigen::Vector3f b = currentPos.dot(m_direction) * m_direction;
    double distance = (currentPos - b).norm();

    return distance;
}

//------------------------------------------------------------------------------------------------------------------------------------------

CartesianVector PlaneModel::GetDirection()
{
    return CartesianVector(m_direction[0], m_direction[1], m_direction[2]);
}

//------------------------------------------------------------------------------------------------------------------------------------------

CartesianVector PlaneModel::GetOrigin()
{
    return CartesianVector(m_origin[0], m_origin[1], m_origin[2]);
}

//------------------------------------------------------------------------------------------------------------------------------------------

virtual void PlaneModel::Initialize(const ParameterVector &inputParams) override
{
    if(inputParams.size() != 3)
        throw std::runtime_error("PlaneModel - Number of input parameters does not match minimum number required for this model.");

    Eigen::Matrix3f m;
    Eigen::Vector3f totals = {0.0, 0.0, 0.0};

    for (auto param : inputParams)
    {
        auto currentPoint = std::dynamic_pointer_cast<Point3D>(param);

        if(currentPoint == nullptr)
            throw std::runtime_error("PlaneModel - inputParams type mismatch. It is not a Point3D.");

        totals += (*currentPoint).m_Point3D;
    }

    m_origin = totals / inputParams.size();

    for (unsigned int i = 0; i < inputParams.size(); ++i)
    {
        auto currentPoint = *std::dynamic_pointer_cast<Point3D>(inputParams[i]);
        Eigen::Vector3f shiftedPoint = Eigen::Vector3f(currentPoint.m_Point3D - m_origin);
        m.row(i) = shiftedPoint;
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinV);
    m_direction = svd.matrixV().col(0); // TODO: Check this, returns matrix not vector.
}

//------------------------------------------------------------------------------------------------------------------------------------------

virtual std::pair<double, ParameterVector> PlaneModel::Evaluate(const ParameterVector &paramsToEval, double threshold) override
{
    ParameterVector inliers;
    float totalParams = paramsToEval.size();

    for(auto& param : paramsToEval)
        if(ComputeDistanceMeasure(param) < threshold)
            inliers.push_back(param);

    double inlierFraction = inliers.size() / totalParams;
    return std::make_pair(inlierFraction, inliers);
}

} // namespace lar_content

#endif // LAR_PLANE_MODEL_RANSAC_H
