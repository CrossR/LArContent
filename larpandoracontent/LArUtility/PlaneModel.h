#pragma once

#include "AbstractModel.hpp"

typedef std::array<GRANSAC::VPFloat, 3> Vector3VP;

class Point3D
: public GRANSAC::AbstractParameter
{
    public:
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

class PlaneModel
: public GRANSAC::AbstractModel<3>
{
    protected:
        // Parametric form
        GRANSAC::VPFloat m_a, m_b, m_c, m_d; // ax + by + cz + d = 0 where n = [a b c] is the normalized normal vector

        virtual GRANSAC::VPFloat ComputeDistanceMeasure(std::shared_ptr<GRANSAC::AbstractParameter> Param) override
        {
            auto ExtPoint3D = std::dynamic_pointer_cast<Point3D>(Param);
            if(ExtPoint3D == nullptr)
                throw std::runtime_error("PlaneModel::ComputeDistanceMeasure() - Passed parameter are not of type Point3D.");

            // Return distance between passed "point" and this line
            GRANSAC::VPFloat Dist = fabs(m_a * (*ExtPoint3D)[0] + m_b * (*ExtPoint3D)[1] + m_c * (*ExtPoint3D)[2] + m_d);

            // // Debug
            // std::cout << "Point: " << ExtPoint3D[0] << ", " << ExtPoint3D[1] << std::endl;
            // std::cout << "Line: " << m_a << " x + " << m_b << " y + "  << m_c << std::endl;
            // std::cout << "Distance: " << Dist << std::endl << std::endl;

            return Dist;
        };

    public:
        PlaneModel(std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> InputParams)
        {
            Initialize(InputParams);
        };

        Vector3VP GetPlaneNormal(void) { return Vector3VP{m_a, m_b, m_c}; };

        virtual void Initialize(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> &InputParams) override
        {
            if(InputParams.size() != 3)
                throw std::runtime_error("PlaneModel - Number of input parameters does not match minimum number required for this model.");

            // Check for AbstractParamter types
            auto Point1 = std::dynamic_pointer_cast<Point3D>(InputParams[0]);
            auto Point2 = std::dynamic_pointer_cast<Point3D>(InputParams[1]);
            auto Point3 = std::dynamic_pointer_cast<Point3D>(InputParams[2]);
            if(Point1 == nullptr || Point2 == nullptr || Point3 == nullptr)
                throw std::runtime_error("PlaneModel - InputParams type mismatch. It is not a Point3D.");

            std::copy(InputParams.begin(), InputParams.end(), m_MinModelParams.begin());

            // Compute the plane parameters
            // Assuming points are not collinear
            pandora::CartesianVector vec1((*Point1)[0], (*Point1)[1], (*Point1)[2]);
            pandora::CartesianVector vec2((*Point2)[0], (*Point2)[1], (*Point2)[2]);
            pandora::CartesianVector vec3((*Point3)[0], (*Point3)[1], (*Point3)[2]);

            pandora::CartesianVector cross = (vec2 - vec1).GetCrossProduct(vec3 - vec1);

            pandora::CartesianVector normal(0.f, 0.f, 0.f);

            if (!(std::fabs(cross.GetMagnitude()) < std::numeric_limits<float>::epsilon()))
                normal = cross.GetUnitVector();

            m_a = normal.GetX();
            m_b = normal.GetY();
            m_c = normal.GetZ();
            m_d = - (m_a * (*Point1)[0] + m_b * (*Point1)[1] + m_c * (*Point1)[2]); // Could be any one of the three points
        };

        virtual std::pair<GRANSAC::VPFloat, std::vector<std::shared_ptr<GRANSAC::AbstractParameter>>> Evaluate(const std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> &EvaluateParams, GRANSAC::VPFloat Threshold) override
        {
            std::vector<std::shared_ptr<GRANSAC::AbstractParameter>> Inliers;
            int nTotalParams = EvaluateParams.size();
            int nInliers = 0;

            for(auto& Param : EvaluateParams)
            {
                if(ComputeDistanceMeasure(Param) < Threshold)
                {
                    Inliers.push_back(Param);
                    nInliers++;
                }
            }

            GRANSAC::VPFloat InlierFraction = GRANSAC::VPFloat(nInliers) / GRANSAC::VPFloat(nTotalParams); // This is the inlier fraction

            return std::make_pair(InlierFraction, Inliers);
        };
};
