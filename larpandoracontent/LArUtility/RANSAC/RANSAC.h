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

            std::shared_ptr<T> m_bestModel; // Pointer to the best model, valid only after Estimate() is called
            ParameterVector m_bestInliers;

            int m_numIterations; // Number of iterations before termination
            double m_threshold; // The threshold for computing model consensus
            std::mutex m_inlierAccumMutex;
            std::vector<std::mt19937> m_randEngines;

            void CheckModel(
                    int threadNumber,
                    std::vector<double> &inlierFrac,
                    std::vector<ParameterVector> &inliers,
                    std::vector<std::shared_ptr<T>> &sampledModels
            )
            {
                int numThreads = std::max(1U, std::thread::hardware_concurrency());
                int i = threadNumber;

                while (i < m_numIterations)
                {
                        // Select t_numParams random samples
                        ParameterVector RandomSamples(t_numParams);
                        ParameterVector RemainderSamples = m_data; // Without the chosen random samples

                        // Shuffle to avoid picking the same element more than once
                        std::shuffle(RemainderSamples.begin(), RemainderSamples.end(), m_randEngines[threadNumber]);
                        std::copy(RemainderSamples.begin(), RemainderSamples.begin() + t_numParams, RandomSamples.begin());

                        std::shared_ptr<T> randomModel = std::make_shared<T>(RandomSamples);

                        // Evaluate the current model, so that its performance can be checked later.
                        std::pair<double, ParameterVector> evalPair = randomModel->Evaluate(RemainderSamples, m_threshold);

                        // Push back into history.
                        std::unique_lock<std::mutex> inlierGate(m_inlierAccumMutex);
                        inliers[i] = evalPair.second;
                        sampledModels[i] = randomModel;
                        inlierGate.unlock();

                        inlierFrac[i] = evalPair.first;
                        i += numThreads;
                }
            }

    public:
            RANSAC(void)
            {
                    int numThreads = std::max(1U, std::thread::hardware_concurrency());
                    std::vector<int> seeds(numThreads);
                    std::iota(seeds.begin(), seeds.end(), 0);
                    for (int i = 0; i < numThreads; ++i)
                    {
                            m_randEngines.push_back(std::mt19937(seeds[i]));
                    }

                    Reset();
            };

            virtual ~RANSAC(void) {};

            void Reset(void) { m_data.clear(); };

            void Initialize(double threshold, int numIterations = 1000)
            {
                    m_threshold = threshold;
                    m_numIterations = numIterations;
            };

            std::shared_ptr<T> GetBestModel(void) { return m_bestModel; };
            const ParameterVector& GetBestInliers(void) { return m_bestInliers; };

            bool Estimate(const ParameterVector &Data)
            {
                    if (Data.size() <= t_numParams)
                            return false;

                    m_data = Data;

                    std::vector<double> inlierFrac(m_numIterations, 0.0);
                    std::vector<ParameterVector> inliers(m_numIterations);
                    std::vector<std::shared_ptr<T>> sampledModels(m_numIterations);

                    int numThreads = std::max(1U, std::thread::hardware_concurrency());
                    std::vector<std::thread> threads;

                    for (int i = 0; i < numThreads; ++i)
                    {
                        std::thread t(&RANSAC::CheckModel, this, i, std::ref(inlierFrac), std::ref(inliers), std::ref(sampledModels));
                        threads.push_back(std::move(t));
                    }

                    for (int i = 0; i < numThreads; ++i)
                        threads[i].join();

                    double bestModelScore = -1;
                    for (int i = 0; i < m_numIterations; ++i)
                    {
                        if (inlierFrac[i] > bestModelScore)
                            {
                                bestModelScore = inlierFrac[i];
                                m_bestModel = sampledModels[i];
                                m_bestInliers = inliers[i];
                            }
                    }

                    Reset();

                    return true;
            };
    };
} // namespace lar_content
#endif // LAR_RANSAC_ALGO_TEMPLATED_H
