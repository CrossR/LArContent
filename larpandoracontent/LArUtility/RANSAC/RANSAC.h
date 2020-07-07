/**
 *  @file   larpandoracontent/LArUtility/RANSAC/RANSAC.h
 *
 *  @brief  Header and implementation for the RANSAC.
 *          Single header file since since that is easier for something so templated.
 *          Code is from https://github.com/drsrinathsridhar/GRANSAC
 *          Original license file can be found in larpandoracontent/LArUtility/RANSAC/LICENSE.
 *
 *  $Log: $
 */

#ifndef LAR_RANSAC_ALGO_TEMPLATED_H
#define LAR_RANSAC_ALGO_TEMPLATED_H 1

#include <mutex>
#include <thread>
#include <random>
#include <vector>

#include "larpandoracontent/LArUtility/RANSAC/AbstractModel.h"

namespace lar_content
{

template <class T, int t_numParams>
class RANSAC
{

public:

    RANSAC(double threshold, int numIterations = 1000)
    {
        this->Reset();

        m_threshold = threshold;
        m_numIterations = numIterations;
    };

    /**
     *  @brief  Reset all the data.
     */
    void Reset() { m_data.clear(); };

    virtual ~RANSAC() {};

    std::shared_ptr<T> GetBestModel() { return m_bestModel; };
    ParameterVector& GetBestInliers() { return m_bestInliers; };

    std::shared_ptr<T> GetSecondBestModel() { return m_secondBestModel; };
    ParameterVector& GetSecondBestInliers() { return m_secondBestInliers; };

//------------------------------------------------------------------------------------------------------------------------------------------

    /**
     *  @brief  Given a vector of data, get the best and second best model for it.
     *
     *  @param  data The data to use for this RANSAC run.
     */
    bool Estimate(const ParameterVector &data)
    {
        if (data.size() <= t_numParams)
            return false;

        m_data = data;
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

        this->Reset();

        return true;
    }

//------------------------------------------------------------------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------------------------------------------------------------------

    /**
     *  @brief  Compare a candidate model to the current best model, to decide
     *  if it is sufficiently unique compared to the current best model, and
     *  the existing most-unique model.
     *
     *  @param  candidateInliers  The current inlying parameters to consider.
     *  @param  diff A parameter vector to fill with unique parameters.
     */
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

//------------------------------------------------------------------------------------------------------------------------------------------

    /**
     *  @brief  Specific implementation of std::uniform_int_distribution.
     *
     *  @param  low  Lowest number in range to pick.
     *  @param  high Highest number in range to pick.
     *  @param  eng The MT19937 to give a random number to scale.
     */
    int uniform_distribution(const int low, const int high, std::mt19937 &eng)
    {
        const double answer = eng() / (1.0 + eng.max());
        const int total_range = high - low + 1;
        return (int) (answer * total_range) + low;
    };

//------------------------------------------------------------------------------------------------------------------------------------------

    /**
     *  @brief  Generate a vector of samples to generate models from. Used to
     *  avoid threading issues if the samples were picked on the fly by each
     *  thread.
     */
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

//------------------------------------------------------------------------------------------------------------------------------------------

    /**
     *  @brief  Generate and check a model based on some samples. Written to be
     *  run across multiple threads simultaneously.
     *
     *  @param  threadNumber  The current thread number, to ensure unique writing of results.
     *  @param  inlierFrac Vector to store the percentage of inlying parameters for each model.
     *  @param  inliers The vector of inlying parameters for each model.
     *  @param  sampledModels Vector of the actual models that were generated and evaluated.
     *  @param  finished An early return flag, set if another thread finds a perfect model.
     */
    void CheckModel(int threadNumber, std::vector<double> &inlierFrac, std::vector<ParameterVector> &inliers,
            std::vector<std::shared_ptr<T>> &sampledModels, bool finished)
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
    };

//------------------------------------------------------------------------------------------------------------------------------------------

};
} // namespace lar_content
#endif // LAR_RANSAC_ALGO_TEMPLATED_H
