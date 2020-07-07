// Code is from https://github.com/drsrinathsridhar/GRANSAC
// Original license file can be found in larpandoracontent/LArUtility/RANSAC/LICENSE.

#ifndef LAR_RANSAC_ALGO_TEMPLATED_H
#define LAR_RANSAC_ALGO_TEMPLATED_H 1

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
    template <class T, int t_numParams>
    class RANSAC
    {
    private:
            ParameterVector m_data;                 ///< The data for the RANSAC model.
            std::vector<ParameterVector> m_samples; ///< Samples to use for model generation.

            std::shared_ptr<T> m_secondBestModel;   ///< Second best model, with most unique parameters compared to first.
            std::shared_ptr<T> m_bestModel;         ///< Pointer to best model, valid only after Estimate() is called.

            ParameterVector m_bestInliers;          ///< The parameters in the best model.
            ParameterVector m_secondBestInliers;    ///< The parameters in the second model.

            int m_secondUniqueParamCount = 0;       ///< A count of how many unique parameters are in the second model.
            int m_numIterations;                    ///< Number of RANSAC iterations.
            double m_threshold;                     ///< Threshold for model consensus.
            std::mutex m_inlierAccumMutex;          ///< Mutex to guard model storing over multiple threads.

// TODO: Move all to implementation file.
            void CompareToBestModel(ParameterVector &candidateInliers, ParameterVector &diff)
            {
                if (candidateInliers.size() < m_secondUniqueParamCount)
                    return;

                for (unsigned int i = 0; i < candidateInliers.size(); ++i)
                {
                    const auto p1 = candidateInliers[i];

                    const int maxDiffSize = diff.size() + (candidateInliers.size() - i);

                    // ATTN: Early return.
                    if (maxDiffSize < m_secondUniqueParamCount)
                        return;

                    // INFO: If current parameter is in the best model, skip it.
                    if (m_bestModel->ComputeDistanceMeasure(p1) < m_threshold)
                        continue;

                    diff.push_back(p1);
                }
            };

            // ATTN: This is equivalent to std::uniform_int_distribution in result, but
            // since the implementation of that differs across compilers, it is hard-coded
            // here.
            int uniform_distribution(const int low, const int high, std::mt19937 &eng) {
                const double answer = eng() / (1.0 + eng.max());
                const int total_range = high - low + 1;
                return (int) (answer * total_range) + low;
            }

            // ATTN: Generate the samples for model generation. Done to prevent randomness issues
            // when calling across multiple threads.
            void GenerateSamples()
            {
                std::mt19937 eng(m_data.size());
                m_samples.clear();

                for (unsigned int i = 0; i < m_numIterations; ++i)
                {
                    ParameterVector currentParameters(t_numParams);

                    for (unsigned int j = 0; j < t_numParams; ++j)
                        currentParameters[j] = m_data[uniform_distribution(0, m_data.size() - 1, eng)];

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
                const int numThreads = std::max(1U, std::thread::hardware_concurrency());
                int i = threadNumber;

                while (i < m_numIterations && !finished)
                {
                    // Evaluate the current model, so that its performance can be checked later.
                    const std::shared_ptr<T> randomModel = std::make_shared<T>(m_samples[i]);
                    const std::pair<double, ParameterVector> evalPair = randomModel->Evaluate(m_data, m_threshold);

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

            std::shared_ptr<T> GetBestModel() { return m_bestModel; };
            ParameterVector& GetBestInliers() { return m_bestInliers; };

            std::shared_ptr<T> GetSecondBestModel() { return m_secondBestModel; };
            ParameterVector& GetSecondBestInliers() { return m_secondBestInliers; };

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
