// Code is from https://github.com/drsrinathsridhar/GRANSAC
// Original license file can be found in larpandoracontent/LArUtility/RANSAC/LICENSE.

#ifndef LAR_RANSAC_ALGO_TEMPLATED_H
#define LAR_RANSAC_ALGO_TEMPLATED_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <random>
#include <vector>

#include "AbstractModel.h"

namespace lar_content
{
    // T - AbstractModel
    template <class T, int t_numParams>
    class RANSAC
    {
    private:
            ParameterVector m_data; // All the data
            std::vector<ParameterVector> m_samples; // Samples to use for model generation.

            std::shared_ptr<T> m_bestModel; // Pointer to the best model, valid only after Estimate() is called
            ParameterVector m_bestInliers;

            std::shared_ptr<T> m_secondBestModel; // Second best model, that is the further away from the best, i.e. most unique parameters.
            ParameterVector m_secondBestInliers;
            int m_secondUniqueParamCount = 0;

            int m_numIterations; // Number of iterations before termination
            double m_threshold; // The threshold for computing model consensus
            std::mutex m_inlierAccumMutex;

            void CompareToBestModel(ParameterVector &candidateInliers, ParameterVector &diff)
            {
                if (candidateInliers.size() < m_secondUniqueParamCount)
                    return;

                for (unsigned int i = 0; i < candidateInliers.size(); ++i)
                {
                    auto p1 = candidateInliers[i];

                    int maxDiffSize = diff.size() + (candidateInliers.size() - i);

                    // If diff can never be bigger than the current best, stop.
                    if (maxDiffSize < m_secondUniqueParamCount)
                        return;

                    // If the current point is an inlier of the best model, skip it.
                    if (m_bestModel->ComputeDistanceMeasure(p1) < m_threshold)
                        continue;

                    diff.push_back(p1);
                }
            };

            void GenerateSamples()
            {
                std::mt19937 eng(m_data.size());
                std::uniform_int_distribution<> distr(0, m_data.size() - 1);
                m_samples.clear();

                for (unsigned int i = 0; i < m_numIterations; ++i)
                {
                    ParameterVector currentParameters(t_numParams);

                    for (unsigned int j = 0; j < t_numParams; ++j)
                        currentParameters[j] = m_data[distr(eng)];

                    m_samples.push_back(currentParameters);
                }
            };

            void CheckModel(
                    int threadNumber,
                    std::vector<double> &inlierFrac,
                    std::vector<ParameterVector> &inliers,
                    std::vector<std::shared_ptr<T>> &sampledModels,
                    bool finished
            )
            {
                int numThreads = std::max(1U, std::thread::hardware_concurrency());
                int i = threadNumber;

                while (i < m_numIterations && !finished)
                {
                    // Evaluate the current model, so that its performance can be checked later.
                    std::shared_ptr<T> randomModel = std::make_shared<T>(m_samples[i]);
                    std::pair<double, ParameterVector> evalPair = randomModel->Evaluate(m_data, m_threshold);

                    // Push back into history.
                    std::unique_lock<std::mutex> inlierGate(m_inlierAccumMutex);
                    inliers[i] = evalPair.second;
                    sampledModels[i] = randomModel;

                    // If a model contained every data point, stop.
                    if (evalPair.first == m_data.size())
                        finished = true;

                    inlierGate.unlock();

                    inlierFrac[i] = evalPair.first;
                    i += numThreads;
                }
            }

    public:
            RANSAC(double threshold, int numIterations = 1000)
            {
                Reset();

                m_threshold = threshold;
                m_numIterations = numIterations;
            };

            virtual ~RANSAC(void) {};

            void Reset(void) { m_data.clear(); };

            std::shared_ptr<T> GetBestModel(void) { return m_bestModel; };
            ParameterVector& GetBestInliers(void) { return m_bestInliers; };

            std::shared_ptr<T> GetSecondBestModel(void) { return m_secondBestModel; };
            ParameterVector& GetSecondBestInliers(void) { return m_secondBestInliers; };

            bool Estimate(const ParameterVector &Data)
            {
                if (Data.size() <= t_numParams)
                    return false;

                m_data = Data;
                this->GenerateSamples();

                std::vector<double> inlierFrac(m_numIterations, 0.0);
                std::vector<ParameterVector> inliers(m_numIterations);
                std::vector<std::shared_ptr<T>> sampledModels(m_numIterations);

                int numThreads = std::max(1U, std::thread::hardware_concurrency());
                std::vector<std::thread> threads;
                bool finished = false;

                for (int i = 0; i < numThreads; ++i)
                {
                    std::thread t(&RANSAC::CheckModel, this,
                            i, std::ref(inlierFrac), std::ref(inliers), std::ref(sampledModels), std::ref(finished));
                    threads.push_back(std::move(t));
                }

                for (int i = 0; i < numThreads; ++i)
                    threads[i].join();

                double bestModelScore = -1;

                for (int i = 0; i < sampledModels.size(); ++i)
                {
                    if (inlierFrac[i] == 0.0)
                        continue;

                    std::cout << ">> Model " << i << " used samples: ";

                    for (auto p : m_samples[i])
                    {
                        auto hit = *std::dynamic_pointer_cast<Point3D>(p);
                        std::cout << hit.m_ProtoHit.GetPosition3D() << ", ";
                    }

                    std::cout << " and had " << inliers[i].size() << " inliers." << std::endl;

                    if (inlierFrac[i] > bestModelScore)
                    {
                        bestModelScore = inlierFrac[i];
                        m_bestModel = sampledModels[i];
                        m_bestInliers = inliers[i];
                    }
                }

                double secondModelScore = -1;

                for (int i = 0; i < sampledModels.size(); ++i)
                {
                    if (inlierFrac[i] == 0.0)
                        continue;

                    if (inlierFrac[i] > (secondModelScore * 0.9))
                    {
                        ParameterVector diff;
                        this->CompareToBestModel(inliers[i], diff);

                        if (diff.size() > 0)
                            std::cout << ">>>>> Diff Size: " << diff.size() << std::endl;

                        if (diff.size() > m_secondUniqueParamCount)
                        {
                            secondModelScore = inlierFrac[i];
                            m_secondUniqueParamCount = diff.size();
                            m_secondBestModel = sampledModels[i];
                            m_secondBestInliers = inliers[i];
                        }
                    }
                }

                Reset();

                return true;
            };
    };
} // namespace lar_content
#endif // LAR_RANSAC_ALGO_TEMPLATED_H
