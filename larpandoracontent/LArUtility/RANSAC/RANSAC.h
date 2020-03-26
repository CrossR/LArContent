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

            std::shared_ptr<T> m_secondBestModel; // Second best model, that is the further away from the best, i.e. most unique parameters.
            ParameterVector m_secondBestInliers;

            int m_numIterations; // Number of iterations before termination
            double m_threshold; // The threshold for computing model consensus
            std::mutex m_inlierAccumMutex;
            std::vector<std::mt19937> m_randEngines;

            void CompareToBestModel(ParameterVector &candidateInliers, ParameterVector &diff)
            {
                for (unsigned int i = 0; i < candidateInliers.size(); ++i)
                {
                    auto p1 = candidateInliers[i];

                    int maxDiffSize = diff.size() + (candidateInliers.size() - i);

                    // If diff can never be bigger than the current best, stop.
                    if (maxDiffSize < m_secondBestInliers.size())
                        return;

                    bool uniqueParameter = true;

                    for (auto p2 : m_bestInliers)
                    {
                        if ((*p1).equals(p2))
                        {
                            uniqueParameter = false;
                            break;
                        }
                    }

                    if (uniqueParameter)
                        diff.push_back(p1);
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

                        // If a model contained every data point, stop.
                        if (evalPair.first == RemainderSamples.size())
                            finished = true;

                        if (finished)
                            break;

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
            ParameterVector& GetBestInliers(void) { return m_bestInliers; };

            std::shared_ptr<T> GetSecondBestModel(void) { return m_secondBestModel; };
            ParameterVector& GetSecondBestInliers(void) { return m_secondBestInliers; };

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
                    unsigned int uniqueParameterCount = 0;

                    for (int i = 0; i < sampledModels.size(); ++i)
                    {
                        if (inlierFrac[i] == 0.0)
                            continue;

                        // TODO: The comparison below is pretty expensive.
                        //       Smarter ways to skip it? Origin, Direction
                        //       and inliers are what we have to use. Also
                        //       any short-circuit logic for the comparison
                        //       itself.

                        if (inlierFrac[i] > (secondModelScore * 0.9))
                        {
                            ParameterVector diff;
                            this->CompareToBestModel(inliers[i], diff);
                            std::cout << ">>>>> Diff Size: " << diff.size() << std::endl;

                            if (diff.size() > uniqueParameterCount)
                            {
                                secondModelScore = inlierFrac[i];
                                uniqueParameterCount = diff.size();
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
