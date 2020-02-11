/**
 *  @file   larpandoracontent/LArUtility/PlaneModel.h
 *
 *  @brief  Header file for the PlaneModel, to be used in RANSAC.
 *
 *  $Log: $
 */
#ifndef LAR_PLANE_MODEL_RANSAC_H
#define LAR_PLANE_MODEL_RANSAC_H

#include "AbstractModel.h"

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

    typedef ThreeDHitCreationAlgorithm::ProtoHit ProtoHit;

    Point3D(ProtoHit* p)
    {
        m_ProtoHit = p;
        m_Point3D(0) = p->GetPosition3D().GetX();
        m_Point3D(1) = p->GetPosition3D().GetY();
        m_Point3D(2) = p->GetPosition3D().GetZ();
    };

    ProtoHit* m_ProtoHit = nullptr;
    Eigen::Vector3f m_Point3D;

    float& operator[](int i)
    {
        if(i < 3)
            return m_Point3D(i);

        throw std::runtime_error("Point3D::Operator[] - Index exceeded bounds.");
    };
};

/**
 *  @brief  Class that implements a PlaneModel, to be fit using RANSAC.
 */
class PlaneModel: public AbstractModel<3>
{
protected:

    Eigen::Vector3f m_direction;
    Eigen::Vector3f m_origin;

    /**
     *  @brief  Project point to line and work out the distance.
     */
    virtual double ComputeDistanceMeasure(SharedParameter param) override
    {
        auto currentPoint = std::dynamic_pointer_cast<Point3D>(param);
        if(currentPoint == nullptr)
            throw std::runtime_error("PlaneModel::ComputeDistanceMeasure() - Passed parameter are not of type Point3D.");

        auto point = *currentPoint;
        point.m_Point3D -= m_origin;

        Eigen::Vector3f b = point.m_Point3D.dot(m_direction) * m_direction;
        double distance = (point.m_Point3D - b).norm();

        return distance;
    };

public:

    PlaneModel(ParameterVector inputParams) { Initialize(inputParams); };

    virtual void Initialize(const ParameterVector &inputParams) override
    {
        if(inputParams.size() != 3)
            throw std::runtime_error("PlaneModel - Number of input parameters does not match minimum number required for this model.");

        Eigen::Matrix3f m;
        Eigen::Vector3f totals = {0.0, 0.0, 0.0};

        // Check for AbstractParamter types, before getting values out to calculate origin.
        for (auto param : inputParams)
        {
            auto currentPoint = std::dynamic_pointer_cast<Point3D>(param);

            if(currentPoint == nullptr)
                throw std::runtime_error("PlaneModel - inputParams type mismatch. It is not a Point3D.");

            totals += (*currentPoint).m_Point3D;
        }

        m_origin = totals / inputParams.size();

        // Use the calculated origin to normalise the data, and also build up
        // the matrix to run SVD (since we are over-constrained).
        for (unsigned int i = 0; i < inputParams.size(); ++i)
        {
            auto currentPoint = *std::dynamic_pointer_cast<Point3D>(inputParams[i]);
            Eigen::Vector3f shiftedPoint = Eigen::Vector3f(currentPoint.m_Point3D - m_origin);
            m.row(i) = shiftedPoint;
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinV);
        m_direction = svd.matrixV();
    };

    virtual std::pair<double, ParameterVector> Evaluate(const ParameterVector &paramsToEval, double threshold) override
    {
        ParameterVector inliers;
        float totalParams = paramsToEval.size();

        for(auto& param : paramsToEval)
            if(ComputeDistanceMeasure(param) < threshold)
                inliers.push_back(param);

        double inlierFraction = inliers.size() / totalParams;
        return std::make_pair(inlierFraction, inliers);
    };
};

} // namespace lar_content

#endif // LAR_PLANE_MODEL_RANSAC_H
