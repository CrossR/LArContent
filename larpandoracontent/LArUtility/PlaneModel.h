/**
 *  @file   larpandoracontent/LArUtility/PlaneModel.h
 *
 *  @brief  Header file for the PlaneModel, to be used in RANSAC.
 *
 *  $Log: $
 */
#ifndef LAR_KD_TREE_LINKER_ALGO_TEMPLATED_H
#define LAR_KD_TREE_LINKER_ALGO_TEMPLATED_H

#include "AbstractModel.hpp"

#include <Eigen/Core>
#include <Eigen/SVD>

namespace lar_content
{

typedef std::array<GRANSAC::VPFloat, 3> Vector3VP;
typedef std::shared_ptr<GRANSAC::AbstractParameter> SharedParameter;
typedef std::vector<SharedParameter> ParameterVector;

/**
 *  @brief  Class that implements a 3D Point in the form RANSAC needs.
 */
class Point3D : public GRANSAC::AbstractParameter
{
public:
    /**
     *  @brief  Default constructor
     */
    Point3D(GRANSAC::VPFloat x, GRANSAC::VPFloat y, GRANSAC::VPFloat z)
    {
        m_Point3D[0] = x;
        m_Point3D[1] = y;
        m_Point3D[2] = z;
    };

    Vector3VP m_Point3D;

    GRANSAC::VPFloat& operator[](int i)
    {
        if(i < 3)
            return m_Point3D[i];

        throw std::runtime_error("Point3D::Operator[] - Index exceeded bounds.");
    };

};

/**
 *  @brief  Class that implements a PlaneModel, to be fit using RANSAC.
 */
class PlaneModel: public GRANSAC::AbstractModel<3>
{
protected:

    // Parametric form
    GRANSAC::VPFloat m_a, m_b, m_c, m_d; // ax + by + cz + d = 0 where n = [a b c] is the normalized normal vector

    virtual GRANSAC::VPFloat ComputeDistanceMeasure(SharedParameter param) override
    {
        auto currentPoint = std::dynamic_pointer_cast<Point3D>(param);
        if(currentPoint == nullptr)
            throw std::runtime_error("PlaneModel::ComputeDistanceMeasure() - Passed parameter are not of type Point3D.");

        // Return distance between passed "point" and this line
        GRANSAC::VPFloat distance = fabs(m_a * (*currentPoint)[0] + m_b * (*currentPoint)[1] + m_c * (*currentPoint)[2] + m_d);

        return distance;
    };

public:
    PlaneModel(ParameterVector inputParams)
    {
        Initialize(inputParams);
    };

    Vector3VP GetPlaneNormal(void) { return Vector3VP{m_a, m_b, m_c}; };

    virtual void Initialize(const ParameterVector &inputParams) override
    {
        if(inputParams.size() != 3)
            throw std::runtime_error("PlaneModel - Number of input parameters does not match minimum number required for this model.");

        // Check for AbstractParamter types
        auto point1 = std::dynamic_pointer_cast<Point3D>(inputParams[0]);
        auto point2 = std::dynamic_pointer_cast<Point3D>(inputParams[1]);
        auto point3 = std::dynamic_pointer_cast<Point3D>(inputParams[2]);
        if(point1 == nullptr || point2 == nullptr || point3 == nullptr)
            throw std::runtime_error("PlaneModel - inputParams type mismatch. It is not a Point3D.");

        // TODO: Perhaps we want to take a mean here?
        Eigen::Matrix3f m;
        ParameterVector normalisedData;

        Vector3VP totals = {0.0, 0.0, 0.0};
        Vector3VP means = {0.0, 0.0, 0.0};

        for (auto param : inputParams)
        {
            auto currentPoint = *std::dynamic_pointer_cast<Point3D>(param);
            totals[0] += currentPoint[0];
            totals[1] += currentPoint[1];
            totals[2] += currentPoint[2];
        }

        means[0] = totals[0] / inputParams.size();
        means[1] = totals[1] / inputParams.size();
        means[2] = totals[2] / inputParams.size();

        for (unsigned int i = 0; i < inputParams.size(); ++i)
        {
            auto param = inputParams[i];
            auto currentPoint = *std::dynamic_pointer_cast<Point3D>(param);
            Point3D shiftedPoint = Point3D(currentPoint[0] - means[0],
                                           currentPoint[1] - means[1],
                                           currentPoint[2] - means[2]
            );

            SharedParameter candidatePoint = std::make_shared<Point3D>(shiftedPoint);
            normalisedData.push_back(candidatePoint);
            m(i, 0) = shiftedPoint[0];
            m(i, 1) = shiftedPoint[1];
            m(i, 2) = shiftedPoint[2];
        }

        Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinV);
        std::copy(normalisedData.begin(), normalisedData.end(), m_MinModelParams.begin());

        m_a = svd.matrixV()(0);
        m_b = svd.matrixV()(1);
        m_c = svd.matrixV()(2);
        m_d = - (m_a * (*point1)[0] + m_b * (*point1)[1] + m_c * (*point1)[2]); // Could be any one of the three points
    };

    virtual std::pair<GRANSAC::VPFloat, ParameterVector> Evaluate(const ParameterVector &paramsToEval, GRANSAC::VPFloat threshold) override
    {
        ParameterVector inliers;
        float totalParams = paramsToEval.size();
        float numInliers = 0.0;

        for(auto& param : paramsToEval)
        {
            if(ComputeDistanceMeasure(param) < threshold)
            {
                inliers.push_back(param);
                numInliers++;
            }
        }

        GRANSAC::VPFloat inlierFraction = GRANSAC::VPFloat(numInliers / totalParams); // This is the inlier fraction

        return std::make_pair(inlierFraction, inliers);
    };
};

} // namespace lar_content

#endif // LAR_KD_TREE_LINKER_ALGO_TEMPLATED_H
